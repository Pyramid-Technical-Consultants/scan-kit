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
from PySide6.QtCore import QEvent, Qt, QTimer, Slot
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
from ..common.user_store import PREF_LAST_DICOM_DIR, prefs_get, prefs_set
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
from .patient_qa_plots import OVERLAY_ALPHA, QaPlots
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

        self._shown_now = None  # _shown() as of the last full redraw; cursor moves reuse it
        self._contours: dict[tuple[int, int, int], np.ndarray] = {}  # (view, slice, roi) -> segments in mm
        self._plots = QaPlots(self._on_pick, self._on_page)
        self._make_3d()
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._plots.native)
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
        self._apply_theme()
        self._redraw()
        if study:
            self.load(study)

    def _apply_theme(self) -> None:
        bg, fg = self.palette().window().color().name(), self.palette().windowText().color().name()
        self._plots.set_theme(bg, fg)
        self._vispy.bgcolor = bg

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, "_plots"):
            self._apply_theme()

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
        folder = QFileDialog.getExistingDirectory(self, "DICOM study folder (CT, RTSTRUCT, RT Ion Plan, RTDOSE)",
                                                  str(prefs_get(PREF_LAST_DICOM_DIR) or ""))
        if folder:
            prefs_set(PREF_LAST_DICOM_DIR, folder)
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
        self._contours.clear()
        (dx, dy, dz), (nx, ny, nz) = grid.spacing, grid.shape
        self._plots.set_extents(((nx * dx, ny * dy), (nx * dx, nz * dz), (ny * dy, nz * dz)),
                                ((dx, dy), (dx, dz), (dy, dz)))
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
        """(volume, colormap name, lo, hi, difference) for the chosen display, or None."""
        mode = self._show.current_key()
        if mode == GAMMA:
            if self._gamma_grid is None:
                return None
            return self._gamma_grid, GAMMA_CMAP, 0.0, 2.0, False
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

    def _checked(self) -> list[int]:
        return [r for r in range(self._roi_list.count()) if self._roi_list.item(r).checkState() == Qt.CheckState.Checked]

    def _redraw(self) -> None:
        if self._grid is None:
            self._shown_now = None
            self._plots.clear("Open a DICOM folder")
            return
        self._shown_now = shown = self._shown()
        self._plots.show_slices()
        self._draw_slices()
        self._draw_colorbar(shown)
        self._draw_dvh()
        self._draw_gamma()
        self._update_3d(shown)

    def _draw_slices(self) -> None:
        shown, grid = self._shown_now, self._grid
        (dx, dy, dz), (nx, ny, nz) = grid.spacing, grid.shape
        i, j, k = self._cursor
        # ponytail: image index order on screen (radiological for HFS); prone or feet-first CTs show flipped.
        cuts = (lambda v: v[k], lambda v: v[:, j, :], lambda v: v[:, :, i])
        index = (k, j, i)
        pixels = ((dx, dy), (dx, dz), (dy, dz))
        size = ((nx * dx, ny * dy), (nx * dx, nz * dz), (ny * dy, nz * dz))
        cross = (((i + 0.5) * dx, (j + 0.5) * dy), ((i + 0.5) * dx, (k + 0.5) * dz), ((j + 0.5) * dy, (k + 0.5) * dz))
        rois = self._case.structures.rois if self._case.structures is not None else ()
        checked = [r for r in self._checked() if r < len(rois)]
        wash = None
        if shown is not None:
            vol, scale, lo, hi, _diff = shown
            lut = (255 * colormap_samples(scale)).astype(np.uint8)
            gamma = self._show.current_key() == GAMMA
            wash = [self._wash(cut(vol), lut, lo, hi, gamma) for cut in cuts]
        lines = []
        for v in range(3):
            (w, h), (cx, cy) = size[v], cross[v]
            segs = [np.array([[cx, 0.0], [cx, h], [0.0, cy], [w, cy]], np.float32)]
            colors = [np.tile(np.float32([1.0, 1.0, 1.0, 0.45]), (4, 1))]
            for r in checked:
                s = self._contour(v, index[v], r, cuts[v], pixels[v])
                segs.append(s)
                colors.append(np.tile(np.float32([*(np.array(rois[r].color) / 255.0), 1.0]), (len(s), 1)))
            lines.append((np.concatenate(segs), np.concatenate(colors)))
        titles = (f"Axial · {k + 1}/{nz}", f"Coronal · {j + 1}/{ny}", f"Sagittal · {i + 1}/{nx}")
        self._plots.set_slices([cut(self._hu) for cut in cuts], wash, lines, titles, CT_WINDOW)

    @staticmethod
    def _wash(img: np.ndarray, lut: np.ndarray, lo: float, hi: float, gamma: bool) -> np.ndarray:
        """RGBA uint8 dose wash: the scale's color, transparent where the dose is faint."""
        t = np.clip((img - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(img)
        rgba = np.empty((*img.shape, 4), np.uint8)
        rgba[..., :3] = lut[(t * (len(lut) - 1)).astype(np.intp)]
        faint = img <= 0 if gamma else np.abs(img) < WASH_FLOOR * max(abs(lo), abs(hi))
        rgba[..., 3] = np.where(faint, 0, round(255 * OVERLAY_ALPHA))
        return rgba

    def _contour(self, view: int, index: int, r: int, cut, pixel) -> np.ndarray:
        """ROI *r*'s outline on one slice as line segments ``(2n, 2)`` in mm, cached per slice."""
        key = (view, index, r)
        if key not in self._contours:
            from contourpy import contour_generator

            mask = cut((self._bits[r // MAX_MASK_BITS] >> np.uint32(r % MAX_MASK_BITS)) & 1)
            segs = [np.zeros((0, 2), np.float32)]
            rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
            if rows.size:
                # Contour a one-voxel-padded crop: outlines close, and cost follows the ROI, not the slice.
                r0, c0 = rows[0] - 1, cols[0] - 1
                crop = np.zeros((rows[-1] - r0 + 2, cols[-1] - c0 + 2), np.float32)
                crop[1:-1, 1:-1] = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
                for line in contour_generator(z=crop).lines(0.5):
                    pts = (line + (c0 + 0.5, r0 + 0.5)) * pixel
                    segs.append(np.repeat(pts, 2, axis=0)[1:-1])
            self._contours[key] = np.concatenate(segs).astype(np.float32)
        return self._contours[key]

    def _draw_colorbar(self, shown) -> None:
        if shown is None:
            self._plots.set_colorbar(None, 0.0, 1.0, "")
            return
        _vol, scale, lo, hi, diff = shown
        mode = self._show.current_key()
        title = "γ" if mode == GAMMA else ("Delivered − planned, Gy(RBE)" if diff else f"{mode.capitalize()} dose, Gy(RBE)")
        self._plots.set_colorbar(colormap_samples(scale), lo, hi, title)

    def _draw_dvh(self) -> None:
        rois = self._case.structures.rois if self._case.structures is not None else ()
        names = {self._roi_list.item(r).text() for r in self._checked()}
        both = DELIVERED in self._dvh
        curves, legend, top = [], [], 0.0
        for kind in (PLANNED, DELIVERED):
            faded = both and kind == PLANNED
            for roi in rois:
                h = self._dvh.get(kind, {}).get(roi.name)
                if h is None or roi.name not in names:
                    continue
                color = (*(np.array(roi.color) / 255.0), 0.5 if faded else 1.0)
                curves.append((h.edges, 100.0 * h.cumulative, color, 1.1 if faded else 1.6))
                top = max(top, float(h.edges[np.flatnonzero(h.cumulative > 0)[-1] + 1]) if h.cumulative.any() else 0.0)
                if not faded:
                    legend.append((roi.name, color))
        self._plots.set_dvh(curves, legend, top * 1.05, "" if curves else "DVH when the run finishes")
        if curves:
            self._plots.set_dvh_title("DVH · Gy(RBE), course" + (" · delivered bright, plan faded" if both else ""))

    def _draw_gamma(self) -> None:
        g = self._gamma
        if g is None:
            self._plots.set_gamma(None, None, "No gamma")
            return
        counts, edges = np.histogram(np.minimum(g.gamma[g.gamma > 0], g.criteria.cap), bins=50,
                                     range=(0.0, g.criteria.cap))
        self._plots.set_gamma(counts, edges,
                              f"γ {g.criteria.dose_pct:g}%/{g.criteria.dta_mm:g}mm · {100 * g.rate:.1f} % pass")

    def _on_pick(self, view: int, x: float, y: float) -> None:
        if self._grid is None:
            return
        (a, b), sp = ((0, 1), (0, 2), (1, 2))[view], self._grid.spacing
        cursor = self._cursor.copy()
        cursor[a], cursor[b] = int(x // sp[a]), int(y // sp[b])
        cursor = np.clip(cursor, 0, np.array(self._grid.shape) - 1)
        if not np.array_equal(cursor, self._cursor):
            self._cursor = cursor
            self._draw_slices()

    def _on_page(self, view: int, steps: int) -> None:
        if self._grid is None:
            return
        axis = (2, 1, 0)[view]
        self._cursor[axis] = int(np.clip(self._cursor[axis] + steps, 0, self._grid.shape[axis] - 1))
        self._draw_slices()

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
        vol, name, lo, hi, diff = shown
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
        from matplotlib.image import imsave

        buf = io.BytesIO()
        imsave(buf, self._plots.canvas.render(), format="png")
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
