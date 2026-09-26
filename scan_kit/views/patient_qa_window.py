"""Patient QA window: planning CT, contours and Monte Carlo dose from DICOM.

The plan is recalculated on the CT; logged sessions add the delivered dose of the
same beams, and a TPS RTDOSE on the frame adds 3D gamma. Tri-planar slices, the
ray-marched 3D dose, DVHs, clinical goals and an HTML/RTDOSE report export.
"""

from __future__ import annotations

import datetime
import html
import io
import logging
import math
import re
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..common.progress_line import ProgressLine
from ..common.segmented_control import SegmentedControl
from ..dicom import StudyIndex, write_dose
from ..dicom.calibration import MCSQUARE_SCANNERS, CtCalibration
from ..dicom.structures import MAX_MASK_BITS, mask_bits
from ..qa import BeamModel, delivery_from_session, fraction_runs, group_fractions, match_delivery, patient_run, plan_spots
from ..qa.analysis import BEAM_SUMMATIONS, RBE, Goal, dvhs, gamma_vs_tps, provenance, resample
from ..qa.beam_model import MCSQUARE_BDL
from ..qa.dose_calc import BODY_HU
from ..qa.report import write_report
from .async_refresh import DebouncedBackgroundTask
from .dose_volume_catalog import DEFAULT_DIVERGENT_SCALE, DEFAULT_MC_HISTORIES, DEFAULT_SCALE, MC_HISTORIES, MC_SEED
from .dose_volume_fill import GAMMA_CMAP, colormap_samples
from .dose_volume_physics import GammaCriteria
from .plot_view_shell import VispyViewWindow, make_side_panel_column, run_view_window

_log = logging.getLogger(__name__)

PLANNED, DELIVERED, DIFF, GAMMA = "planned", "delivered", "diff", "gamma"
PREVIEW_S = 0.25
PREVIEW_SHARE = 0.2  # most of the UI thread live previews may take from the Monte Carlo
SLICE_S = 0.03
CT_WINDOW = (-500.0, 500.0)  # HU shown black to white
WASH_FLOOR = 0.1  # dose below this share of the maximum is not washed
DEFAULT_BDL = "BDL_default_DN_RangeShifter"
GAMMA_CRITERIA = ((3.0, 3.0), (3.0, 2.0), (2.0, 2.0), (1.0, 1.0))
_LABEL_WIDTH = 90


def load_study(folder: str, session_ids: Sequence[str], base_dir: str):
    """(PatientCase, [Delivery]) for *folder* and the sessions that have spot logs."""
    case = StudyIndex.scan(folder).load()
    logs = [d for d in (delivery_from_session(s, base_dir) for s in session_ids) if d is not None]
    return case, logs


def _guarded(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - the window reports it
        _log.exception("patient QA load failed")
        return exc


class PatientQaWindow(VispyViewWindow):
    def __init__(
        self, session_ids: Sequence[str], base_dir: str, *, study: str | None = None, parent: QWidget | None = None,
    ) -> None:
        super().__init__(title="Patient QA (DICOM)", side_panel_default_width=360, parent=parent)
        self._session_ids, self._base_dir = list(session_ids), base_dir
        self._case = None
        self._logs = []  # Delivery per logged session
        self._plan = None
        self._fractions = []  # [[(session label, DeliveryMatch)]] of the logs that fit the plan
        self._picked = []  # the (label, DeliveryMatch) being transported
        self._tps = []  # TPS RTDOSEs for the plan
        self._runs: dict[str, object] = {}
        self._grid = None
        self._doses: dict[str, np.ndarray] = {}
        self._n_fractions = 1
        self._hu = None
        self._bits: list[np.ndarray] = []
        self._cursor = np.zeros(3, dtype=int)  # i, j, k on the dose grid
        self._gamma = None
        self._gamma_grid = None
        self._dvh: dict[str, dict] = {}
        self._goal_results = []
        self._preview_at = 0.0
        self._preview_cost = 0.0
        self._run_histories = None
        self._closed = False
        self._updating = True

        self.figure = Figure(figsize=(9, 8), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        gs = self.figure.add_gridspec(2, 3, height_ratios=(3, 2))
        self._ax_axial, self._ax_coronal, self._ax_sagittal = (self.figure.add_subplot(gs[0, c]) for c in range(3))
        self._ax_dvh = self.figure.add_subplot(gs[1, :2])
        self._ax_gamma = self.figure.add_subplot(gs[1, 2])
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._make_3d()
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.canvas)
        split.addWidget(self._vispy.native)
        split.setSizes([900, 600])
        self._plot_layout.addWidget(split, 1)
        self.progress = ProgressLine(self._plot_host)
        self._timer = QTimer(self)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._tick)
        self._goal_timer = QTimer(self)
        self._goal_timer.setSingleShot(True)
        self._goal_timer.setInterval(400)
        self._goal_timer.timeout.connect(self._update_goals)
        self.set_side_panel(self._build_controls())
        self._loader = DebouncedBackgroundTask(debounce_ms=0, parent=self)
        self._loader.finished.connect(self._on_loaded)
        self._updating = False
        self._redraw()
        if study:
            self.load(study)

    # ---- controls -------------------------------------------------------------------------------

    def _build_controls(self) -> QWidget:
        panel, layout = make_side_panel_column()
        study = self._group(layout, "Study")
        button = QPushButton("Open DICOM folder…")
        button.clicked.connect(self._pick_study)
        study.addWidget(button)
        self._study_label = QLabel("No study loaded")
        self._study_label.setWordWrap(True)
        study.addWidget(self._study_label)
        self._plan_combo = self._combo(study, "Plan", self._on_plan_changed)
        self._beam_combo = self._combo(study, "Beams", self._restart)
        self._fraction_combo = self._combo(study, "Fraction", self._restart)
        self._delivery_label = QLabel("")
        self._delivery_label.setWordWrap(True)
        study.addWidget(self._delivery_label)

        mc = self._group(layout, "Monte Carlo")
        self._histories = SegmentedControl([(str(h), f"{h / 1e6:g}M") for h in MC_HISTORIES])
        self._histories.set_current(str(DEFAULT_MC_HISTORIES))
        self._histories.selectionChanged.connect(lambda key: None if key == self._run_histories else self._restart())
        self._row(mc, "Histories", self._histories)
        self._scanner_combo = self._combo(mc, "CT curve", self._restart)
        for p in sorted(MCSQUARE_SCANNERS.iterdir()) if MCSQUARE_SCANNERS.is_dir() else ():
            self._scanner_combo.addItem(p.name, p.name)
        self._bdl_combo = self._combo(mc, "Beam model", self._restart)
        for p in sorted(MCSQUARE_BDL.glob("*.txt")):
            self._bdl_combo.addItem(p.stem.removeprefix("BDL_"), p.stem)
        self._bdl_combo.setCurrentIndex(max(self._bdl_combo.findData(DEFAULT_BDL), 0))
        self._mc_label = QLabel("")
        self._mc_label.setWordWrap(True)
        mc.addWidget(self._mc_label)

        show = self._group(layout, "Display")
        self._show = SegmentedControl([(PLANNED, "Plan"), (DELIVERED, "Delivered"), (DIFF, "Δ"), (GAMMA, "γ")])
        self._show.set_current(PLANNED)
        self._show.selectionChanged.connect(self._on_show_changed)
        self._row(show, "Show", self._show)
        self._roi_list = QListWidget()
        self._roi_list.setMinimumHeight(120)
        self._roi_list.itemChanged.connect(self._on_rois_changed)
        show.addWidget(self._roi_list)

        gamma = self._group(layout, "Gamma vs TPS")
        self._tps_combo = self._combo(gamma, "TPS dose", self._update_gamma)
        self._criteria_combo = self._combo(gamma, "Criteria", self._update_gamma)
        for d, r in GAMMA_CRITERIA:
            self._criteria_combo.addItem(f"{d:g} % / {r:g} mm", (d, r))
        self._gamma_label = QLabel("")
        self._gamma_label.setWordWrap(True)
        gamma.addWidget(self._gamma_label)

        goals = self._group(layout, "Clinical goals")
        self._rx = QDoubleSpinBox()
        self._rx.setRange(0.0, 200.0)
        self._rx.setDecimals(2)
        self._rx.setSuffix(" Gy(RBE)")
        self._rx.valueChanged.connect(lambda _v: self._goal_timer.start())
        self._row(goals, "Rx (course)", self._rx)
        self._goals_edit = QPlainTextEdit()
        self._goals_edit.setPlaceholderText("PTV: D95% >= 95%\nCord: Dmax < 45 Gy\nLung: V20Gy < 30%")
        self._goals_edit.setFixedHeight(90)
        self._goals_edit.textChanged.connect(self._goal_timer.start)
        goals.addWidget(self._goals_edit)
        self._goals_label = QLabel("")
        self._goals_label.setWordWrap(True)
        self._goals_label.setTextFormat(Qt.TextFormat.RichText)
        goals.addWidget(self._goals_label)

        self._export_button = QPushButton("Export report…")
        self._export_button.clicked.connect(self._pick_export)
        self._export_button.setEnabled(False)
        layout.addWidget(self._export_button)
        layout.addStretch(1)
        return panel

    @staticmethod
    def _group(layout: QVBoxLayout, title: str) -> QVBoxLayout:
        group = QGroupBox(title)
        inner = QVBoxLayout(group)
        inner.setContentsMargins(8, 12, 8, 8)
        inner.setSpacing(6)
        layout.addWidget(group)
        return inner

    @staticmethod
    def _row(layout: QVBoxLayout, label: str, widget: QWidget) -> None:
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        text = QLabel(label)
        text.setFixedWidth(_LABEL_WIDTH)
        row.addWidget(text)
        row.addWidget(widget, 1)
        layout.addWidget(host)

    def _combo(self, layout: QVBoxLayout, label: str, handler) -> QComboBox:
        combo = QComboBox()
        combo.currentIndexChanged.connect(lambda _i: None if self._updating else handler())
        self._row(layout, label, combo)
        return combo

    # ---- loading --------------------------------------------------------------------------------

    def _pick_study(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "DICOM study folder (CT, RTSTRUCT, RT Ion Plan, RTDOSE)")
        if folder:
            self.load(folder)

    def load(self, folder: str) -> None:
        sessions, base = list(self._session_ids), self._base_dir
        self._study_label.setText(f"Reading {folder}…")
        self.progress.busy()
        self._loader.schedule(lambda: _guarded(lambda: load_study(folder, sessions, base)))

    @Slot(int, object)
    def _on_loaded(self, gen: int, result) -> None:
        if gen != self._loader.generation or self._closed:
            return
        if not isinstance(result, tuple):
            self.progress.done()
            self._study_label.setText(f"Could not load the study: {result}")
            return
        self._stop()
        self._case, self._logs = result
        self._plan, self._grid, self._fractions, self._picked, self._tps = None, None, [], [], []
        case = self._case
        rois = case.structures.rois if case.structures is not None else ()
        self._study_label.setText(
            f"CT {case.ct.grid.shape[0]}×{case.ct.grid.shape[1]}×{case.ct.grid.shape[2]}, "
            f"{len(rois)} structures, {len(case.plans)} plan(s), {len(case.doses)} dose(s); "
            f"{len(self._logs)} of {len(self._session_ids)} session(s) logged spots",
        )
        self._updating = True
        for combo in (self._plan_combo, self._beam_combo, self._fraction_combo, self._tps_combo):
            combo.clear()
        for i, p in enumerate(case.plans):
            self._plan_combo.addItem(p.label or f"Plan {i + 1}", i)
        self._roi_list.clear()
        for roi in rois:
            item = QListWidgetItem(roi.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if roi.kind != "EXTERNAL" else Qt.CheckState.Unchecked)
            self._roi_list.addItem(item)
        self._updating = False
        if not case.plans:
            self.progress.done()
            self._study_label.setText(self._study_label.text() + ". No RT Ion Plan on this frame.")
            self._redraw()
            return
        self._on_plan_changed()

    def _on_plan_changed(self) -> None:
        self._stop()
        plan = self._case.plans[int(self._plan_combo.currentData() or 0)]
        try:
            matched = [(d.label, m) for d in self._logs if len((m := match_delivery(plan, d)).spots)]
        except Exception as exc:  # noqa: BLE001 - shown, not raised into Qt
            self._fail("Cannot match the spot logs to this plan", exc)
            return
        self._plan = plan
        label_of = {id(m): label for label, m in matched}
        self._fractions = [[(label_of[id(m)], m) for m in f] for f in group_fractions([m for _l, m in matched])]
        self._updating = True
        self._beam_combo.clear()
        self._beam_combo.addItem("All beams", None)
        for b in plan.beams:
            self._beam_combo.addItem(f"{b.number} · {b.name} (G{b.gantry:g}° C{b.couch:g}°)", b.number)
        self._fraction_combo.clear()
        if self._fractions:
            for k, f in enumerate(self._fractions):
                self._fraction_combo.addItem(f"Fraction {k + 1}: beams {', '.join(str(m.beam) for _l, m in f)}", k)
            if len(self._fractions) > 1:
                self._fraction_combo.addItem(f"All {len(self._fractions)} fractions", -1)
        else:
            self._fraction_combo.addItem("Plan only (no spot logs)", None)
        self._tps_combo.clear()
        doses = self._case.doses_for(plan) or [d for d in self._case.doses if not d.plan_uid]
        self._tps_combo.addItem("None", None)
        for i, d in enumerate(doses):
            beam = f" beam {d.beam_number}" if d.summation.upper() in BEAM_SUMMATIONS else ""
            self._tps_combo.addItem(f"{d.summation or 'Dose'}{beam} …{d.sop_uid[-8:]}", i)
        self._tps = doses
        whole = [i for i, d in enumerate(doses) if d.summation.upper() not in BEAM_SUMMATIONS]
        self._tps_combo.setCurrentIndex(whole[0] + 1 if whole else 0)
        self._rx.setValue(plan.prescription if math.isfinite(plan.prescription) else 0.0)
        targets = [r.name for r in (self._case.structures.rois if self._case.structures else ())
                   if r.kind in ("PTV", "CTV", "GTV")]
        if targets and not self._goals_edit.toPlainText().strip():
            self._goals_edit.setPlainText(f"{targets[0]}: D95% >= 95%\n{targets[0]}: D2% <= 107%")
        self._show.set_current(DELIVERED if self._fractions else PLANNED)
        self._updating = False
        self._restart()

    # ---- Monte Carlo ----------------------------------------------------------------------------

    def _selected(self):
        """(label, DeliveryMatch) of the chosen fraction(s) and beams, and how many fractions that is."""
        if not self._fractions:
            return [], 1
        k = self._fraction_combo.currentData()
        chosen = self._fractions if k == -1 else [self._fractions[int(k or 0)]]
        beam = self._beam_combo.currentData()
        per = [[(label, m) for label, m in f if beam is None or m.beam == beam] for f in chosen]
        return [lm for f in per for lm in f], max(sum(1 for f in per if f), 1)

    def _transported_beams(self) -> set:
        if self._picked:
            return {m.beam for _l, m in self._picked}
        beam = self._beam_combo.currentData()
        return {b.number for b in self._plan.beams} if beam is None else {beam}

    def _stop(self) -> None:
        """Drop the runs and everything computed from them."""
        self._timer.stop()
        self._goal_timer.stop()
        for run in self._runs.values():
            run.close()
        self._runs, self._doses, self._dvh, self._gamma, self._gamma_grid = {}, {}, {}, None, None
        self._goal_results = []
        for label in (self._gamma_label, self._goals_label, self._mc_label, self._delivery_label):
            label.setText("")
        self._export_button.setEnabled(False)

    def _fail(self, what: str, exc: Exception) -> None:
        _log.error("%s", what, exc_info=exc)
        self._stop()
        self._mc_label.setText(f"{what}: {exc}")
        self.progress.done()
        self._redraw()

    def _restart(self, *_args) -> None:
        if self._updating or self._plan is None:
            return
        self._stop()
        plan, case = self._plan, self._case
        self._run_histories = self._histories.current_key() or str(DEFAULT_MC_HISTORIES)
        kw = dict(histories=int(self._run_histories), seed=MC_SEED)
        picked, self._n_fractions = self._selected()
        self._picked = picked
        try:
            cal = CtCalibration.mcsquare(self._scanner_combo.currentData() or "default")
            model = BeamModel.read(self._bdl_combo.currentData() or DEFAULT_BDL)
            self._cal, self._model = cal, model
            if picked:
                d, p, grid = fraction_runs(case.ct, cal, model, plan, [m for _l, m in picked], **kw)
                self._runs = {DELIVERED: d, PLANNED: p}
            else:
                beam = self._beam_combo.currentData()
                run, grid = patient_run(case.ct, cal, model, plan,
                                        plan_spots(plan, None if beam is None else {beam}), **kw)
                self._runs = {PLANNED: run}
            if self._grid is None or grid.shape != self._grid.shape or not np.allclose(grid.origin, self._grid.origin):
                self._set_grid(grid)
        except Exception as exc:  # noqa: BLE001 - shown, not raised into Qt
            self._fail("Cannot run", exc)
            return
        self._delivery_label.setText("; ".join(
            f"beam {m.beam}: {m.delivered_mu / m.planned_mu:.1%} MU, {m.position_rms:.2f} mm RMS"
            + (f", {m.unmatched} stray" if m.unmatched else "") for _l, m in picked if m.planned_mu > 0))
        self._show.set_option_enabled(DELIVERED, bool(picked))
        self._show.set_option_enabled(DIFF, bool(picked))
        if not picked and self._show.current_key() in (DELIVERED, DIFF):
            self._show.set_current(PLANNED)
        self._preview_at = 0.0
        self._timer.start()
        self.progress.set_progress(0.0)
        self._redraw()

    def _set_grid(self, grid) -> None:
        self._grid = grid
        ct = self._case.ct
        off = np.rint(ct.grid.to_index(grid.origin)).astype(int)
        nx, ny, nz = grid.shape
        self._hu = ct.hu[off[2]:off[2] + nz, off[1]:off[1] + ny, off[0]:off[0] + nx]
        # Shown and gamma-evaluated dose stays in the patient: air voxels (1 mg/cm³) spike honestly but read as noise.
        # ponytail: gas below BODY_HU inside the patient is blanked too; a BODY contour mask would keep it.
        self._body = self._hu > BODY_HU
        rois = self._case.structures.rois if self._case.structures is not None else ()
        self._bits = [mask_bits(rois[s:s + MAX_MASK_BITS], grid) for s in range(0, len(rois), MAX_MASK_BITS)]
        iso = grid.to_index(self._plan.beams[0].isocenter)
        self._cursor = np.clip(np.rint(iso).astype(int), 0, np.array(grid.shape) - 1)
        self._set_box(grid)

    def _tick(self) -> None:
        try:
            self._advance()
        except Exception as exc:  # noqa: BLE001 - a lost device would otherwise raise every tick
            self._fail("Monte Carlo failed", exc)

    def _advance(self) -> None:
        busy = [r for r in self._runs.values() if not r.done]
        for run in busy:
            run.step(SLICE_S)
        done = all(r.done for r in self._runs.values())
        now = time.perf_counter()
        if not done and now - self._preview_at < max(PREVIEW_S, self._preview_cost / PREVIEW_SHARE):
            return
        self._preview_at = now
        for kind, run in self._runs.items():
            dose = run.preview()
            if dose is not None:
                self._doses[kind] = np.where(self._body, dose * np.float32(RBE), np.float32(0.0))
        progress = float(np.mean([r.progress for r in self._runs.values()]))
        self.progress.set_progress(progress)
        main = self._runs[self._main_kind()]
        self._mc_label.setText(f"{main.histories / 1e6:g}M histories per dose, {progress:.0%}"
                               + (f", ±{100 * main.result.uncertainty:.1f} %" if done else ""))
        if done:
            self._timer.stop()
            self.progress.done()
            self._finish()
        self._redraw()
        self._preview_cost = time.perf_counter() - now

    def _main_kind(self) -> str:
        return DELIVERED if DELIVERED in self._runs else PLANNED

    def _course_scale(self) -> float:
        """Runs hold the selected fractions; DVHs and goals are for the whole course."""
        return self._plan.fractions / max(self._n_fractions, 1)

    def _finish(self) -> None:
        rois = self._case.structures.rois if self._case.structures is not None else ()
        s = self._course_scale() * RBE
        # Unmasked: the air blanking is for display, a DVH counts every voxel of its structure.
        self._dvh = {k: dvhs(r.dose * s, self._grid, rois, bits=self._bits) for k, r in self._runs.items()} if rois else {}
        for run in self._runs.values():
            run.close()
        self._update_gamma(redraw=False)
        self._update_goals()
        self._export_button.setEnabled(True)

    # ---- analysis -------------------------------------------------------------------------------

    def _update_gamma(self, *_args, redraw: bool = True) -> None:
        self._gamma, self._gamma_grid = None, None
        i = self._tps_combo.currentData()
        dose = self._doses.get(self._main_kind())
        done = self._runs and all(r.done for r in self._runs.values())
        self._show.set_option_enabled(GAMMA, i is not None)
        tps = self._tps[int(i)] if i is not None else None
        covers = set()
        if tps is not None:
            per_beam = tps.summation.upper() in BEAM_SUMMATIONS
            covers = {tps.beam_number} if per_beam else {b.number for b in self._plan.beams}
        if tps is None or dose is None or not done:
            self._gamma_label.setText("" if tps is not None else "No TPS dose selected")
        elif covers != self._transported_beams():
            self._gamma_label.setText(f"This TPS dose covers beam(s) {', '.join(map(str, sorted(covers, key=str)))}; "
                                      "transport the same beams to compare")
        else:
            d, r = self._criteria_combo.currentData() or GAMMA_CRITERIA[0]
            try:
                body = resample(self._body.astype(np.float32), self._grid, tps.grid) > 0.5
                self._gamma = gamma_vs_tps(tps, dose / self._n_fractions, self._grid, GammaCriteria(d, r, 10.0),
                                           fractions=self._plan.fractions, mask=body, effective=True)
                self._gamma_grid = resample(self._gamma.gamma, tps.grid, self._grid)
                self._gamma_label.setText(
                    f"{100 * self._gamma.rate:.2f} % pass ({self._gamma.evaluated} voxels), "
                    f"{self._main_kind()} vs TPS, one fraction")
            except Exception as exc:  # noqa: BLE001
                _log.exception("gamma failed")
                self._gamma_label.setText(f"Gamma failed: {exc}")
        if redraw:
            self._redraw()

    def _update_goals(self) -> None:
        dvh = self._dvh.get(self._main_kind(), {})
        rx = self._rx.value() or None
        lines, self._goal_results = [], []
        for text in self._goals_edit.toPlainText().splitlines():
            if not text.strip() or text.lstrip().startswith("#"):
                continue
            try:
                goal = Goal.parse(text)
                if goal.roi not in dvh:
                    raise ValueError(f"no structure {goal.roi!r}" if dvh else "waiting for the dose")
                value, ok = goal.check(dvh[goal.roi], rx)
            except ValueError as exc:
                lines.append(f"<span style='color:gray'>{html.escape(text)}: {html.escape(str(exc))}</span>")
                continue
            self._goal_results.append((goal, value, ok))
            mark = "<b style='color:#2a2'>pass</b>" if ok else "<b style='color:#c22'>FAIL</b>"
            lines.append(f"{mark} {html.escape(goal.text)} ({value:.2f} {html.escape(goal.unit)})")
        self._goals_label.setText("<br>".join(lines))

    # ---- drawing --------------------------------------------------------------------------------

    def _on_show_changed(self, *_args) -> None:
        self._redraw()

    def _on_rois_changed(self, *_args) -> None:
        if not self._updating:
            self._redraw()

    def _shown(self):
        """(volume, colormap, lo, hi, difference) for the chosen display, or None."""
        mode = self._show.current_key()
        if mode == GAMMA:
            if self._gamma_grid is None:
                return None
            return self._gamma_grid, ListedColormap(colormap_samples(GAMMA_CMAP)), 0.0, 2.0, False
        if mode == DIFF:
            d, p = self._doses.get(DELIVERED), self._doses.get(PLANNED)
            if d is None or p is None:
                return None
            reach = float(np.abs(d - p).max()) or 1.0
            return d - p, DEFAULT_DIVERGENT_SCALE, -reach, reach, True
        dose = self._doses.get(mode if mode in self._doses else PLANNED)
        if dose is None:
            return None
        return dose, DEFAULT_SCALE, 0.0, float(dose.max()) or 1.0, False

    def _redraw(self) -> None:
        for ax in (self._ax_axial, self._ax_coronal, self._ax_sagittal, self._ax_dvh, self._ax_gamma):
            ax.cla()
        if self._grid is None:
            self._ax_axial.set_title("Open a DICOM folder")
            for ax in (self._ax_axial, self._ax_coronal, self._ax_sagittal):
                ax.set_axis_off()
            self.canvas.draw_idle()
            return
        shown = self._shown()
        self._draw_slices(shown)
        self._draw_dvh()
        self._draw_gamma()
        self.canvas.draw_idle()
        self._update_3d(shown)

    def _draw_slices(self, shown) -> None:
        dx, dy, dz = self._grid.spacing
        nx, ny, nz = self._grid.shape
        i, j, k = self._cursor
        # ponytail: image index order on screen (radiological for HFS); prone or feet-first CTs show flipped.
        views = (
            (self._ax_axial, lambda v: v[k], (0, nx * dx, ny * dy, 0), "upper", (i + 0.5) * dx, (j + 0.5) * dy),
            (self._ax_coronal, lambda v: v[:, j, :], (0, nx * dx, 0, nz * dz), "lower", (i + 0.5) * dx, (k + 0.5) * dz),
            (self._ax_sagittal, lambda v: v[:, :, i], (0, ny * dy, 0, nz * dz), "lower", (j + 0.5) * dy, (k + 0.5) * dz),
        )
        rois = self._case.structures.rois if self._case.structures is not None else ()
        checked = [r for r in range(min(len(rois), self._roi_list.count()))
                   if self._roi_list.item(r).checkState() == Qt.CheckState.Checked]
        for (ax, cut, extent, origin, cx, cy), name in zip(views, ("Axial", "Coronal", "Sagittal")):
            ax.imshow(cut(self._hu), cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], extent=extent,
                      origin=origin, interpolation="bilinear")
            if shown is not None:
                vol, cmap, lo, hi, diff = shown
                img = cut(vol)
                gamma = self._show.current_key() == GAMMA
                faint = img <= 0 if gamma else np.abs(img) < WASH_FLOOR * max(abs(lo), abs(hi))
                ax.imshow(np.ma.masked_where(faint, img), cmap=cmap, vmin=lo, vmax=hi, alpha=0.55, extent=extent,
                          origin=origin, interpolation="bilinear")
            for r in checked:
                mask = (cut(self._bits[r // MAX_MASK_BITS]) >> np.uint32(r % MAX_MASK_BITS)) & 1
                if mask.any():
                    ax.contour(mask, levels=[0.5], colors=[np.array(rois[r].color) / 255.0], linewidths=0.9,
                               extent=extent, origin=origin)
            ax.axvline(cx, color="w", lw=0.4, alpha=0.5)
            ax.axhline(cy, color="w", lw=0.4, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name, fontsize=9)
        if shown is not None:
            _vol, _cmap, lo, hi, diff = shown
            unit = "γ" if self._show.current_key() == GAMMA else "Gy(RBE)"
            self._ax_axial.set_title(f"Axial · {lo:.3g} to {hi:.3g} {unit}", fontsize=9)

    def _draw_dvh(self) -> None:
        ax = self._ax_dvh
        rois = self._case.structures.rois if self._case.structures is not None else ()
        names = {self._roi_list.item(r).text() for r in range(self._roi_list.count())
                 if self._roi_list.item(r).checkState() == Qt.CheckState.Checked}
        for kind, style in ((DELIVERED, "-"), (PLANNED, "--" if DELIVERED in self._dvh else "-")):
            for roi in rois:
                h = self._dvh.get(kind, {}).get(roi.name)
                if h is not None and roi.name in names:
                    ax.plot(h.edges, 100.0 * h.cumulative, style, color=np.array(roi.color) / 255.0, lw=1.2,
                            label=roi.name if style == "-" else None)
        ax.set_xlabel("Dose (Gy(RBE), course)" + (" — delivered solid, plan dashed" if DELIVERED in self._dvh else ""))
        ax.set_ylabel("Volume (%)")
        ax.set_ylim(0, 102)
        ax.set_xlim(left=0)
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=7, loc="upper right")
        else:
            ax.text(0.5, 0.5, "DVH when the run finishes", ha="center", va="center", transform=ax.transAxes)

    def _draw_gamma(self) -> None:
        ax = self._ax_gamma
        g = self._gamma
        if g is None:
            ax.text(0.5, 0.5, "No gamma", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        vals = g.gamma[g.gamma > 0]
        ax.hist(np.minimum(vals, g.criteria.cap), bins=50, range=(0, g.criteria.cap), color="#4a8")
        ax.axvline(1.0, color="#c22", lw=1)
        ax.set_title(f"γ {g.criteria.dose_pct:g}%/{g.criteria.dta_mm:g}mm: {100 * g.rate:.1f} %", fontsize=9)
        ax.set_yticks([])

    def _on_click(self, event) -> None:
        if self._grid is None or event.xdata is None:
            return
        dx, dy, dz = self._grid.spacing
        axes = {self._ax_axial: ((0, dx), (1, dy)), self._ax_coronal: ((0, dx), (2, dz)),
                self._ax_sagittal: ((1, dy), (2, dz))}
        if event.inaxes not in axes:
            return
        (a, sa), (b, sb) = axes[event.inaxes]
        self._cursor[a] = int(event.xdata // sa)
        self._cursor[b] = int(event.ydata // sb)
        self._cursor = np.clip(self._cursor, 0, np.array(self._grid.shape) - 1)
        self._redraw()

    def _on_scroll(self, event) -> None:
        axis = {self._ax_axial: 2, self._ax_coronal: 1, self._ax_sagittal: 0}.get(event.inaxes)
        if self._grid is None or axis is None:
            return
        self._cursor[axis] = int(np.clip(self._cursor[axis] + (1 if event.button == "up" else -1),
                                         0, self._grid.shape[axis] - 1))
        self._redraw()

    # ---- 3D -------------------------------------------------------------------------------------

    def _make_3d(self) -> None:
        from vispy import scene

        from .dose_volume_raycast import make_dose_box_node
        from .vispy_plot import ensure_gl_plus, make_scene_canvas

        self._vispy = make_scene_canvas(keys="interactive", size=(600, 600), gl="gl+" if ensure_gl_plus() else None)
        self._view = self._vispy.central_widget.add_view()
        self._view.camera = scene.cameras.TurntableCamera(fov=45, elevation=20, azimuth=30)
        self._box = make_dose_box_node()(parent=self._view.scene)
        self._box.transform = scene.transforms.STTransform()
        self._box.set_interp("linear")
        self._textures = None
        self._marching = False
        self._broken = False  # the ray march failed once (no GL 4.3); don't retry every redraw
        self._vispy.on_draw = self._on_draw_3d

    def _set_box(self, grid) -> None:
        from .dose_volume_raycast import _alloc_texture

        size = np.array(grid.shape, dtype=float) * grid.spacing
        self._box.set_box((0.0, 0.0, 0.0), grid.shape, 1.0)
        self._box.transform.scale = tuple(grid.spacing)
        self._box.transform.translate = tuple(-0.5 * size)
        self._textures = (_alloc_texture(grid.shape_zyx), _alloc_texture(grid.shape_zyx))
        self._box.set_volumes(*self._textures)
        half = 0.5 * size
        self._view.camera.set_range(x=(-half[0], half[0]), y=(-half[1], half[1]), z=(-half[2], half[2]))

    def _update_3d(self, shown) -> None:
        if self._textures is None or shown is None:
            self._marching = False
            self._vispy.update()
            return
        vol, cmap, lo, hi, diff = shown
        name = GAMMA_CMAP if isinstance(cmap, ListedColormap) else cmap
        if diff:
            self._textures[0].set_data(np.ascontiguousarray(self._doses[DELIVERED], dtype=np.float32))
            self._textures[1].set_data(np.ascontiguousarray(self._doses[PLANNED], dtype=np.float32))
            peak = float(self._doses[PLANNED].max()) or 1.0
        else:
            self._textures[0].set_data(np.ascontiguousarray(vol, dtype=np.float32))
            peak = hi
        self._box.set_display(gain=1.0, typical=peak, error_scale=0.05, difference=diff, absolute=False, ray=1.0,
                              ray_scale=peak, scale_name=name, lo=lo, hi=hi)
        self._marching = not self._broken
        self._vispy.update()

    def _on_draw_3d(self, event) -> None:
        from vispy.scene import SceneCanvas

        SceneCanvas.on_draw(self._vispy, event)
        if not self._marching:
            return
        try:
            self._box.draw_volume(self._vispy)
        except Exception:  # noqa: BLE001 - the slices still work without GL 4.3
            _log.exception("patient QA ray march failed")
            self._marching = False
            self._broken = True

    # ---- export ---------------------------------------------------------------------------------

    def _pick_export(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export the QA report into")
        if not folder:
            return
        try:
            path = self.export_report(folder)
        except Exception as exc:  # noqa: BLE001 - shown, not raised into Qt
            _log.exception("patient QA export failed")
            self._mc_label.setText(f"Export failed: {exc}")
            return
        self._mc_label.setText(f"Report written to {path}")

    def export_report(self, folder: str | Path) -> Path:
        """RTDOSE of each dose plus an HTML report and DVH CSV in *folder*; returns the report."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self._goal_timer.stop()
        self._update_goals()
        plan = self._plan
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", plan.label or "plan").strip("_") or "plan"
        base = name = f"qa_{stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        n = 1
        while (folder / f"{name}.html").exists():
            n += 1
            name = f"{base}_{n}"
        whole = self._transported_beams() == {b.number for b in plan.beams}
        summation = "RECORD" if self._n_fractions > 1 else "FRACTION_SESSION" if whole else "BEAM_SESSION"
        labels = [label for label, _m in self._picked]
        provs = {}
        for kind, run in self._runs.items():
            provs[kind] = provenance(
                kind=kind, plan=plan, calibration=self._cal, model=self._model, result=run.result,
                histories=run.histories, seed=MC_SEED, deliveries=labels if kind == DELIVERED else (), case=self._case,
            )
            provs[kind]["fractions_summed"] = self._n_fractions
            provs[kind]["dose"] = f"dose-to-water, Gy over {self._n_fractions} fraction(s)"
            write_dose(folder / f"{name}_{kind}.dcm", run.dose, self._grid, self._case.ct, plan_uid=plan.sop_uid,
                       summation=summation, description=f"scan-kit MC {kind}", provenance=provs[kind])
        buf = io.BytesIO()
        self.figure.savefig(buf, format="png", dpi=110)
        main = self._main_kind()
        title = f"Patient QA: {plan.label or 'plan'}, {self._fraction_combo.currentText()}, {main} dose"
        return write_report(folder, name, title=title, provenance=provs[main], dvhs=self._dvh.get(main, {}),
                            goals=self._goal_results, gamma=self._gamma, matches=[m for _l, m in self._picked],
                            png=buf.getvalue())

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt
        self._closed = True
        self._stop()
        super().closeEvent(event)


def run_patient_qa_window(session_ids: Sequence[str], base_dir: str = "test_data") -> None:
    run_view_window(lambda: PatientQaWindow(session_ids, base_dir), maximize=True)
