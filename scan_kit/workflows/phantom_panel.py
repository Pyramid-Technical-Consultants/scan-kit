"""Phantom Synthesis panel: synthetic patient studies (CT, RTSTRUCT, RT Ion Plan, RTDOSE) as real DICOM."""

from __future__ import annotations

import datetime
import logging
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtCore import QEvent, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..common.progress_line import ProgressLine
from ..common.qt_widgets import configure_pane_scroll_area, make_pane_scroll_area, set_pane_scroll_widget
from ..common.user_store import PREF_LAST_DICOM_DIR, prefs_get, prefs_set
from ..dicom import synthetic
from .plan_synthesis.param_form import ParamFormWidget
from .plan_synthesis.params import ParamSpec

_log = logging.getLogger(__name__)

FOV_MARGIN_MM = (40.0, 40.0, 20.0)  # CT field of view beyond the water box, x y z
REFERENCE_SEED = 20260926  # not the QA window's seed, so gamma compares independent noise
REFERENCE_BDL = "BDL_default_DN_RangeShifter"

SPECS = [
    ParamSpec("phantom", "Phantom", "choice", "slabs", field_set="phantom",
              choices=tuple((k, label) for k, (label, _inserts) in synthetic.PHANTOMS.items())),
    ParamSpec("position", "Patient position", "button_group", "HFS", field_set="phantom",
              choices=(("HFS", "HFS"), ("HFP", "HFP"), ("FFS", "FFS"), ("FFP", "FFP"))),
    ParamSpec("pixel", "Pixel spacing", "float", 2.0, minimum=0.5, maximum=5.0, decimals=2, step=0.25,
              suffix="mm", field_set="ct"),
    ParamSpec("slice", "Slice thickness", "float", 2.0, minimum=0.5, maximum=5.0, decimals=2, step=0.5,
              suffix="mm", field_set="ct"),
    ParamSpec("energies", "Energies", "energy_multiselect", [140.0, 130.0, 120.0], field_set="energy"),
    ParamSpec("gantry", "Gantry", "float", 0.0, minimum=0.0, maximum=359.9, decimals=1, step=15.0,
              suffix="°", field_set="geometry"),
    ParamSpec("couch", "Couch", "float", 0.0, minimum=-90.0, maximum=90.0, decimals=1, step=15.0,
              suffix="°", field_set="geometry"),
    ParamSpec("spot_pitch", "Spot pitch", "float", 6.0, minimum=1.0, maximum=12.0, decimals=1, step=1.0,
              suffix="mm", field_set="geometry"),
    ParamSpec("range_shifter_wet", "Range shifter WET (0 = none)", "float", 0.0, minimum=0.0, maximum=100.0,
              decimals=1, step=5.0, suffix="mm", field_set="geometry"),
    ParamSpec("mu_per_spot", "MU per spot", "float", 0.02, minimum=0.001, maximum=10.0, decimals=3, step=0.01,
              field_set="weight"),
    ParamSpec("fractions", "Fractions", "int", 1, minimum=1, maximum=40, field_set="weight"),
    ParamSpec("reference", "Monte Carlo RTDOSE", "bool", True, field_set="dose"),
    ParamSpec("histories", "Histories", "choice", 2_000_000, field_set="dose", visible_when={"reference": (True,)},
              choices=((1_000_000, "1M"), (2_000_000, "2M"), (10_000_000, "10M"))),
]


def ct_size(pixel: float, slice_mm: float) -> tuple[int, int, int]:
    """CT matrix (nx, ny, nz) covering the water box plus :data:`FOV_MARGIN_MM`."""
    return tuple(math.ceil((synthetic.BOX_MM + m) / d) for m, d in zip(FOV_MARGIN_MM, (pixel, pixel, slice_mm)))


def phantom_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """:func:`~scan_kit.dicom.synthetic.write_phantom` keywords for the form's *params*."""
    if not params["energies"]:
        raise ValueError("pick at least one energy")
    return dict(
        phantom=params["phantom"], position=params["position"],
        spacing=(params["pixel"], params["pixel"], params["slice"]), size=ct_size(params["pixel"], params["slice"]),
        energies=tuple(sorted(params["energies"])), gantry=params["gantry"], couch=params["couch"],
        spot_pitch=params["spot_pitch"], range_shifter_wet=params["range_shifter_wet"],
        mu_per_spot=params["mu_per_spot"], fractions=params["fractions"],
    )


def write_study(folder: str | Path, params: dict[str, Any]) -> str:
    """Write the study for the form's *params* into *folder*; returns a one-line note on what was written."""
    ph = synthetic.write_phantom(folder, **phantom_kwargs(params))
    note = f"{len(ph.ct_files)} CT slices, RTSTRUCT and RT Ion Plan"
    if not params["reference"]:
        return note
    from ..gpu import GpuUnavailable

    try:
        write_reference_dose(ph.folder, int(params["histories"]))
    except GpuUnavailable as exc:
        return f"{note}; no reference RTDOSE ({exc})"
    return f"{note} and a Monte Carlo reference RTDOSE"


def write_reference_dose(folder: Path, histories: int) -> Path:
    """The plan's course dose by the Monte Carlo, as an RTDOSE standing in for the TPS's."""
    from ..dicom import StudyIndex, write_dose
    from ..dicom.calibration import CtCalibration
    from ..qa import BeamModel, patient_run, plan_spots

    case = StudyIndex.scan(folder).load()
    plan = case.plan()
    run, grid = patient_run(case.ct, CtCalibration.mcsquare("default"), BeamModel.read(REFERENCE_BDL), plan,
                            plan_spots(plan), histories=histories, seed=REFERENCE_SEED)
    try:
        run.step()
        return write_dose(folder / "RD_reference.dcm", run.dose * plan.fractions, grid, case.ct,
                          plan_uid=plan.sop_uid, description="scan-kit Monte Carlo reference")
    finally:
        run.close()


class PhantomSynthesisPanel(QWidget):
    """Parameters (left) | axial preview and Write (right)."""

    _written = Signal(object, object)  # folder, note or exception

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._preview_timer = QTimer(self, singleShot=True, interval=150)
        self._busy = False
        self._folder: Path | None = None
        self._written.connect(self._on_written)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter()
        root.addWidget(splitter)

        params_host = QWidget()
        params_l = QVBoxLayout(params_host)
        params_l.setContentsMargins(4, 4, 4, 4)
        scroll = make_pane_scroll_area()
        params_l.addWidget(scroll)
        self._form = ParamFormWidget(SPECS, {s.key: s.default for s in SPECS})
        set_pane_scroll_widget(scroll, self._form, host=params_host)
        configure_pane_scroll_area(scroll, host=params_host)
        splitter.addWidget(params_host)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(4, 4, 4, 4)
        actions = QHBoxLayout()
        self._write_btn = QPushButton("Write DICOM…")
        self._write_btn.setToolTip("Write the study into a new folder; open it in Dose Volume as a DICOM study")
        self._write_btn.clicked.connect(self._pick_folder)
        self._show_btn = QPushButton("Show Folder")
        self._show_btn.setEnabled(False)
        self._show_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._folder))))
        actions.addWidget(self._write_btn)
        actions.addWidget(self._show_btn)
        actions.addStretch(1)
        right_l.addLayout(actions)
        self._figure = Figure(figsize=(5, 5), layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._progress = ProgressLine(self._canvas)
        right_l.addWidget(self._canvas, stretch=1)
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        right_l.addWidget(self._summary)
        self._status = QLabel("Synthetic studies carry no patient data. Open one in Dose Volume with Source: DICOM study.")
        self._status.setWordWrap(True)
        right_l.addWidget(self._status)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 760])

        self._preview_timer.timeout.connect(self._draw_preview)
        for editor in self._form.findChildren(QAbstractSpinBox):
            editor.valueChanged.connect(self._preview_timer.start)
        for editor in self._form.findChildren(QComboBox):
            editor.currentIndexChanged.connect(self._preview_timer.start)
        for editor in self._form.findChildren(QRadioButton):
            editor.toggled.connect(self._preview_timer.start)
        for editor in self._form.findChildren(QListWidget):
            editor.itemSelectionChanged.connect(self._preview_timer.start)
        self._draw_preview()

    def _draw_preview(self) -> None:
        from ..dicom.plan import gantry_to_patient

        p = self._form.read_params()
        half = synthetic.BOX_MM / 2 + FOV_MARGIN_MM[0] / 2
        axis = np.linspace(-half, half, 241)
        x, y = np.meshgrid(axis, axis)
        hu = synthetic.phantom_hu(x, y, np.zeros_like(x), p["phantom"])
        self._figure.clear()
        ax = self._figure.add_subplot()
        ax.imshow(hu, cmap="gray", vmin=-1000, vmax=1000, extent=(-half, half, half, -half))
        (x0, x1), (y0, y1), _z = synthetic.PTV
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", lw=1.2))
        ax.text(x1 + 2, y0, "PTV", color="red", fontsize=8, va="top")
        iso = np.array([0.0, (y0 + y1) / 2])
        beam = -gantry_to_patient(p["gantry"], p["couch"], p["position"])[:2, 2]  # projected onto the slice
        if np.hypot(*beam) > 1e-6:
            u, edge = beam / np.hypot(*beam), half - 5.0
            # Back from the isocenter along the beam to the image edge.
            reach = min((iso[i] + edge) / u[i] if u[i] > 0 else (iso[i] - edge) / u[i]
                        for i in range(2) if abs(u[i]) > 1e-9)
            ax.annotate("", iso, iso - reach * u, arrowprops=dict(arrowstyle="->", color="#4da3ff", lw=1.6))
        ax.set_xlabel("x (mm, patient left →)")
        ax.set_ylabel("y (mm, posterior ↓)")
        ax.set_title("Axial slice through the isocenter", fontsize=10)
        bg, fg = self.palette().window().color().name(), self.palette().windowText().color().name()
        self._figure.set_facecolor(bg)
        ax.tick_params(colors=fg)
        for item in (ax.title, ax.xaxis.label, ax.yaxis.label):
            item.set_color(fg)
        for spine in ax.spines.values():
            spine.set_color(fg)
        self._canvas.draw_idle()
        nx, ny, nz = ct_size(p["pixel"], p["slice"])
        spots = len(np.arange(-12.0, 12.01, p["spot_pitch"])) ** 2 * max(len(p["energies"]), 1)
        self._summary.setText(
            f"CT {nx}×{ny}×{nz} voxels at {p['pixel']:g}×{p['pixel']:g}×{p['slice']:g} mm; "
            f"{len(p['energies'])} layers, {spots} spots, {spots * p['mu_per_spot']:.3g} MU per fraction.")

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._preview_timer.start()

    def _pick_folder(self) -> None:
        if self._busy:
            return
        start = str(prefs_get(PREF_LAST_DICOM_DIR) or "")
        parent = QFileDialog.getExistingDirectory(self, "Write the synthetic study into a new folder under",
                                                  str(Path(start).parent) if start else "")
        if parent:
            params = self._form.read_params()
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.write(Path(parent) / f"synthetic_{params['phantom']}_{stamp}", params)

    def write(self, folder: Path, params: dict[str, Any]) -> None:
        """Write the study in the background; :attr:`_status` reports how it went."""
        self._busy = True
        self._write_btn.setEnabled(False)
        self._status.setText(f"Writing {folder}…")
        self._progress.busy()

        def work() -> None:
            try:
                self._written.emit(folder, write_study(folder, params))
            except Exception as exc:  # noqa: BLE001 - reported in the panel
                _log.exception("synthetic study failed")
                self._written.emit(folder, exc)

        threading.Thread(target=work, name="phantom-synthesis", daemon=True).start()

    def _on_written(self, folder: Path, outcome: object) -> None:
        self._busy = False
        self._write_btn.setEnabled(True)
        self._progress.done()
        if isinstance(outcome, Exception):
            self._status.setText(f"Could not write the study: {outcome}")
            return
        self._folder = folder
        self._show_btn.setEnabled(True)
        prefs_set(PREF_LAST_DICOM_DIR, str(folder))
        self._status.setText(f"Wrote {outcome} to {folder}. Dose Volume's Open DICOM folder… starts here next.")
