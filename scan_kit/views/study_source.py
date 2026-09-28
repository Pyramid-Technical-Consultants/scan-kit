"""DICOM study source for the dose views: the plan and logged deliveries through the Monte Carlo on the planning CT.

:class:`StudySource` owns its controls (study, Monte Carlo, structures, TPS reference,
clinical goals, export) and the runs and analysis behind them. It draws nothing: a view
reads :meth:`StudySource.frame` and the doses, and listens to its signals.
"""

from __future__ import annotations

import datetime
import html
import logging
import math
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
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
    QVBoxLayout,
    QWidget,
)

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
from .dose_panes import DoseFrame
from .dose_volume_catalog import DEFAULT_MC_HISTORIES, MC_HISTORIES, MC_SEED
from .dose_volume_physics import GammaCriteria

_log = logging.getLogger(__name__)

PLANNED, DELIVERED = "planned", "delivered"
LABELS = {PLANNED: "Plan", DELIVERED: "Delivered"}
PREVIEW_S = 0.25
PREVIEW_SHARE = 0.2  # most of the UI thread live previews may take from the Monte Carlo
SLICE_S = 0.03
DEFAULT_BDL = "BDL_default_DN_RangeShifter"
PATIENT_AXES = ("X (mm)", "Y (mm)", "Z (mm)")
_LABEL_WIDTH = 100


def load_study(folder: str, session_ids: Sequence[str], base_dir: str):
    """(PatientCase, [Delivery]) for *folder* and the sessions that have spot logs."""
    case = StudyIndex.scan(folder).load()
    logs = [d for d in (delivery_from_session(s, base_dir) for s in session_ids) if d is not None]
    return case, logs


def _guarded(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - the source reports it
        _log.exception("DICOM study load failed")
        return exc


def _group(layout: QVBoxLayout, title: str) -> QVBoxLayout:
    group = QGroupBox(title)
    inner = QVBoxLayout(group)
    inner.setContentsMargins(8, 12, 8, 8)
    inner.setSpacing(6)
    layout.addWidget(group)
    return inner


def _row(layout: QVBoxLayout, label: str, widget: QWidget) -> None:
    host = QWidget()
    row = QHBoxLayout(host)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    text = QLabel(label)
    text.setFixedWidth(_LABEL_WIDTH)
    row.addWidget(text)
    row.addWidget(widget, 1)
    layout.addWidget(host)


def _wrapped(layout: QVBoxLayout) -> QLabel:
    label = QLabel("")
    label.setWordWrap(True)
    layout.addWidget(label)
    return label


class StudySource(QObject):
    """A DICOM study, its plan recalculated and the chosen logged fractions transported.

    Signals: *frameChanged* for a new grid, *dosesChanged* as the Monte Carlo refines or
    finishes, *analysisChanged* when DVHs, gamma or goals change, *roisChanged* when the
    outlined structures change, and *progress* with the share done (None when idle, -1 busy).
    """

    frameChanged = Signal()
    dosesChanged = Signal()
    analysisChanged = Signal()
    roisChanged = Signal()
    progress = Signal(object)

    def __init__(self, session_ids: Sequence[str], base_dir: str, *, criteria: Callable[[], GammaCriteria],
                 snapshot: Callable[[], bytes] | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._session_ids, self._base_dir = list(session_ids), base_dir
        self._criteria, self._snapshot = criteria, snapshot
        self._case = None
        self._logs = []  # Delivery per logged session
        self._plan = None
        self._fractions = []  # [[(session label, DeliveryMatch)]] of the logs that fit the plan
        self._picked = []  # the (label, DeliveryMatch) being transported
        self._tps = []  # TPS RTDOSEs for the plan
        self._runs: dict[str, object] = {}
        self._grid = None
        self._frame: DoseFrame | None = None
        self.doses: dict[str, np.ndarray] = {}  # kind -> Gy(RBE) over the selected fractions, blanked outside the body
        self._n_fractions = 1
        self._body = None
        self.gamma = None
        self.gamma_grid = None  # γ resampled onto the dose grid
        self.dvh: dict[str, dict] = {}
        self._goal_results = []
        self._preview_at = 0.0
        self._preview_cost = 0.0
        self._run_histories = None
        self._closed = False
        self._updating = True
        self._timer = QTimer(self)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._tick)
        self._goal_timer = QTimer(self)
        self._goal_timer.setSingleShot(True)
        self._goal_timer.setInterval(400)
        self._goal_timer.timeout.connect(self._update_goals)
        self._loader = DebouncedBackgroundTask(debounce_ms=0, parent=self)
        self._loader.finished.connect(self._on_loaded)
        self.panel = self._build_controls()
        self._updating = False

    # ---- controls -------------------------------------------------------------------------------

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        study = _group(layout, "Study")
        button = QPushButton("Open DICOM folder…")
        button.clicked.connect(self._pick_study)
        study.addWidget(button)
        self._study_label = _wrapped(study)
        self._study_label.setText("No study loaded")
        self._plan_combo = self._combo(study, "Plan", self._on_plan_changed)
        self._beam_combo = self._combo(study, "Beams", self._restart)
        self._fraction_combo = self._combo(study, "Fraction", self._restart)
        self._dose_choice = SegmentedControl([(PLANNED, "Plan"), (DELIVERED, "Delivered")])
        self._dose_choice.set_current(PLANNED)
        self._dose_choice.setToolTip("The dose drawn, and what the plan is subtracted from.")
        self._dose_choice.selectionChanged.connect(lambda _k: self.dosesChanged.emit())
        _row(study, "Dose", self._dose_choice)
        self._delivery_label = _wrapped(study)

        mc = _group(layout, "Monte Carlo")
        self._histories = SegmentedControl([(str(h), f"{h / 1e6:g}M") for h in MC_HISTORIES])
        self._histories.set_current(str(DEFAULT_MC_HISTORIES))
        self._histories.selectionChanged.connect(lambda key: None if key == self._run_histories else self._restart())
        _row(mc, "Histories", self._histories)
        self._scanner_combo = self._combo(mc, "CT curve", self._restart)
        for p in sorted(MCSQUARE_SCANNERS.iterdir()) if MCSQUARE_SCANNERS.is_dir() else ():
            self._scanner_combo.addItem(p.name, p.name)
        self._bdl_combo = self._combo(mc, "Beam model", self._restart)
        for p in sorted(MCSQUARE_BDL.glob("*.txt")):
            self._bdl_combo.addItem(p.stem.removeprefix("BDL_"), p.stem)
        self._bdl_combo.setCurrentIndex(max(self._bdl_combo.findData(DEFAULT_BDL), 0))
        self._mc_label = _wrapped(mc)

        rois = _group(layout, "Structures")
        self._roi_list = QListWidget()
        self._roi_list.setMinimumHeight(110)
        self._roi_list.itemChanged.connect(lambda _item: None if self._updating else self.roisChanged.emit())
        rois.addWidget(self._roi_list)

        tps = _group(layout, "Gamma vs TPS")
        self._tps_combo = self._combo(tps, "TPS dose", self.update_gamma)
        self._gamma_label = _wrapped(tps)

        goals = _group(layout, "Clinical goals")
        self._rx = QDoubleSpinBox()
        self._rx.setRange(0.0, 200.0)
        self._rx.setDecimals(2)
        self._rx.setSuffix(" Gy(RBE)")
        self._rx.valueChanged.connect(lambda _v: self._goal_timer.start())
        _row(goals, "Rx (course)", self._rx)
        self._goals_edit = QPlainTextEdit()
        self._goals_edit.setPlaceholderText("PTV: D95% >= 95%\nCord: Dmax < 45 Gy\nLung: V20Gy < 30%")
        self._goals_edit.setFixedHeight(90)
        self._goals_edit.textChanged.connect(self._goal_timer.start)
        goals.addWidget(self._goals_edit)
        self._goals_label = _wrapped(goals)
        self._goals_label.setTextFormat(Qt.TextFormat.RichText)

        self._export_button = QPushButton("Export report…")
        self._export_button.clicked.connect(self._pick_export)
        self._export_button.setEnabled(False)
        layout.addWidget(self._export_button)
        return panel

    def _combo(self, layout: QVBoxLayout, label: str, handler) -> QComboBox:
        combo = QComboBox()
        combo.currentIndexChanged.connect(lambda _i: None if self._updating else handler())
        _row(layout, label, combo)
        return combo

    # ---- what a view reads ----------------------------------------------------------------------

    def frame(self) -> DoseFrame | None:
        return self._frame

    @property
    def iso_index(self) -> np.ndarray | None:
        """The first beam's isocenter as an x, y, z index on the dose grid."""
        if self._grid is None or self._plan is None:
            return None
        return self._grid.to_index(self._plan.beams[0].isocenter)

    @property
    def main_kind(self) -> str:
        """The dose drawn: delivered when logs were transported and it is picked, else planned."""
        return DELIVERED if DELIVERED in self._runs and self._dose_choice.current_key() == DELIVERED else PLANNED

    @property
    def title(self) -> str:
        return f"{self._plan.label or 'plan'}, {self._fraction_combo.currentText()}" if self._plan else "DICOM study"

    @property
    def done(self) -> bool:
        return bool(self._runs) and all(r.done for r in self._runs.values())

    def checked(self) -> list[int]:
        return [r for r in range(self._roi_list.count()) if self._roi_list.item(r).checkState() == Qt.CheckState.Checked]

    # ---- loading --------------------------------------------------------------------------------

    def _pick_study(self) -> None:
        folder = QFileDialog.getExistingDirectory(self.panel, "DICOM study folder (CT, RTSTRUCT, RT Ion Plan, RTDOSE)",
                                                  str(prefs_get(PREF_LAST_DICOM_DIR) or ""))
        if folder:
            prefs_set(PREF_LAST_DICOM_DIR, folder)
            self.load(folder)

    def load(self, folder: str) -> None:
        sessions, base = list(self._session_ids), self._base_dir
        self._study_label.setText(f"Reading {folder}…")
        self.progress.emit(-1)
        self._loader.schedule(lambda: _guarded(lambda: load_study(folder, sessions, base)))

    @Slot(int, object)
    def _on_loaded(self, gen: int, result) -> None:
        if gen != self._loader.generation or self._closed:
            return
        if not isinstance(result, tuple):
            self.progress.emit(None)
            self._study_label.setText(f"Could not load the study: {result}")
            return
        self.stop()
        self._case, self._logs = result
        self._plan, self._grid, self._frame, self._fractions, self._picked, self._tps = None, None, None, [], [], []
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
            self.progress.emit(None)
            self._study_label.setText(self._study_label.text() + ". No RT Ion Plan on this frame.")
            self.frameChanged.emit()
            return
        self._on_plan_changed()

    def _on_plan_changed(self) -> None:
        self.stop()
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
        self._dose_choice.set_current(DELIVERED if self._fractions else PLANNED)
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

    def stop(self) -> None:
        """Drop the runs and everything computed from them."""
        self._timer.stop()
        self._goal_timer.stop()
        for run in self._runs.values():
            run.close()
        self._runs, self.doses, self.dvh, self.gamma, self.gamma_grid = {}, {}, {}, None, None
        self._goal_results = []
        for label in (self._gamma_label, self._goals_label, self._mc_label, self._delivery_label):
            label.setText("")
        self._export_button.setEnabled(False)

    def _fail(self, what: str, exc: Exception) -> None:
        _log.error("%s", what, exc_info=exc)
        self.stop()
        self._mc_label.setText(f"{what}: {exc}")
        self.progress.emit(None)
        self.dosesChanged.emit()

    def _restart(self, *_args) -> None:
        if self._updating or self._plan is None:
            return
        self.stop()
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
        self._dose_choice.set_option_enabled(DELIVERED, bool(picked))
        if not picked:
            self._dose_choice.set_current(PLANNED)
        self._preview_at = 0.0
        self._timer.start()
        self.progress.emit(0.0)
        self.dosesChanged.emit()

    def _set_grid(self, grid) -> None:
        self._grid = grid
        ct = self._case.ct
        off = np.rint(ct.grid.to_index(grid.origin)).astype(int)
        nx, ny, nz = grid.shape
        hu = ct.hu[off[2]:off[2] + nz, off[1]:off[1] + ny, off[0]:off[0] + nx]
        # Shown and gamma-evaluated dose stays in the patient: air voxels (1 mg/cm³) spike honestly but read as noise.
        # ponytail: gas below BODY_HU inside the patient is blanked too; a BODY contour mask would keep it.
        self._body = hu > BODY_HU
        rois = self._case.structures.rois if self._case.structures is not None else ()
        bits = [mask_bits(rois[s:s + MAX_MASK_BITS], grid) for s in range(0, len(rois), MAX_MASK_BITS)]
        sp = np.asarray(grid.spacing, dtype=float)
        # IEC gantry Z points at the source, so the beam travels along −Z; in grid axes, like the transport.
        direction = grid.axes.T @ (self._plan.beams[0].gantry_to_patient() @ np.array([0.0, 0.0, -1.0]))
        # ponytail: shown in index order, radiological for HFS; prone or feet-first CTs show flipped.
        self._frame = DoseFrame(
            origin=np.asarray(grid.corner, dtype=float), spacing=sp, shape=tuple(grid.shape),
            names=PATIENT_AXES, beam_dir=direction, ct=hu, rois=[(r.name, tuple(r.color)) for r in rois],
            bits=bits, flip_axial=True,
        )
        self.frameChanged.emit()

    def _tick(self) -> None:
        try:
            self._advance()
        except Exception as exc:  # noqa: BLE001 - a lost device would otherwise raise every tick
            self._fail("Monte Carlo failed", exc)

    def _advance(self) -> None:
        busy = [r for r in self._runs.values() if not r.done]
        for run in busy:
            run.step(SLICE_S)
        done = self.done
        now = time.perf_counter()
        if not done and now - self._preview_at < max(PREVIEW_S, self._preview_cost / PREVIEW_SHARE):
            return
        self._preview_at = now
        for kind, run in self._runs.items():
            dose = run.preview()
            if dose is not None:
                self.doses[kind] = np.where(self._body, dose * np.float32(RBE), np.float32(0.0))
        progress = float(np.mean([r.progress for r in self._runs.values()]))
        main = self._runs[self.main_kind]
        self._mc_label.setText(f"{main.histories / 1e6:g}M histories per dose, {progress:.0%}"
                               + (f", ±{100 * main.result.uncertainty:.1f} %" if done else ""))
        if done:
            self._timer.stop()
            self.progress.emit(None)
            self._finish()
        else:
            self.progress.emit(progress)
        self.dosesChanged.emit()
        self._preview_cost = time.perf_counter() - now

    def _course_scale(self) -> float:
        """Runs hold the selected fractions; DVHs and goals are for the whole course."""
        return self._plan.fractions / max(self._n_fractions, 1)

    def _finish(self) -> None:
        rois = self._case.structures.rois if self._case.structures is not None else ()
        s = self._course_scale() * RBE
        # Unmasked: the air blanking is for display, a DVH counts every voxel of its structure.
        bits = self._frame.bits
        self.dvh = {k: dvhs(r.dose * s, self._grid, rois, bits=bits) for k, r in self._runs.items()} if rois else {}
        for run in self._runs.values():
            run.close()
        self.update_gamma(notify=False)
        self._update_goals()
        self._export_button.setEnabled(True)
        self.analysisChanged.emit()

    # ---- analysis -------------------------------------------------------------------------------

    def update_gamma(self, *_args, notify: bool = True) -> None:
        """Gamma of the drawn dose against the chosen TPS dose, with the view's criteria."""
        self.gamma, self.gamma_grid = None, None
        i = self._tps_combo.currentData()
        dose = self.doses.get(self.main_kind)
        tps = self._tps[int(i)] if i is not None else None
        covers = set()
        if tps is not None:
            per_beam = tps.summation.upper() in BEAM_SUMMATIONS
            covers = {tps.beam_number} if per_beam else {b.number for b in self._plan.beams}
        if tps is None or dose is None or not self.done:
            self._gamma_label.setText("" if tps is not None else "No TPS dose selected")
        elif covers != self._transported_beams():
            self._gamma_label.setText(f"This TPS dose covers beam(s) {', '.join(map(str, sorted(covers, key=str)))}; "
                                      "transport the same beams to compare")
        else:
            crit = self._criteria()
            try:
                body = resample(self._body.astype(np.float32), self._grid, tps.grid) > 0.5
                self.gamma = gamma_vs_tps(tps, dose / self._n_fractions, self._grid, crit,
                                          fractions=self._plan.fractions, mask=body, effective=True)
                self.gamma_grid = resample(self.gamma.gamma, tps.grid, self._grid)
                self._gamma_label.setText(
                    f"{100 * self.gamma.rate:.2f} % pass ({self.gamma.evaluated} voxels), "
                    f"{LABELS[self.main_kind].lower()} vs TPS, one fraction")
            except Exception as exc:  # noqa: BLE001
                _log.exception("gamma failed")
                self._gamma_label.setText(f"Gamma failed: {exc}")
        if notify:
            self.analysisChanged.emit()

    def _update_goals(self) -> None:
        dvh = self.dvh.get(self.main_kind, {})
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

    # ---- export ---------------------------------------------------------------------------------

    def _pick_export(self) -> None:
        folder = QFileDialog.getExistingDirectory(self.panel, "Export the QA report into")
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
        main = self.main_kind
        title = f"Patient QA: {plan.label or 'plan'}, {self._fraction_combo.currentText()}, {LABELS[main].lower()} dose"
        return write_report(folder, name, title=title, provenance=provs[main], dvhs=self.dvh.get(main, {}),
                            goals=self._goal_results, gamma=self.gamma, matches=[m for _l, m in self._picked],
                            png=self._snapshot() if self._snapshot else None)

    def close(self) -> None:
        self._closed = True
        self.stop()
