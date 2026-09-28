"""Qt shell for the 3D dose-volume viewer (visPy)."""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..common import ViewSettings
from ..common.progress_line import ProgressLine
from ..common.segmented_control import SegmentedControl
from ..common.plotting import format_session_legend_label
from ..common.session_notes import load_notes
from .async_refresh import DebouncedBackgroundTask
from .dose_volume_catalog import (
    DEFAULT_DIVERGENT_SCALE,
    DEFAULT_ERROR_MODE,
    DEFAULT_ERROR_MU,
    DEFAULT_ERROR_PCT,
    DEFAULT_ERROR_SCALE,
    DEFAULT_GAIN,
    DEFAULT_GANTRY_DEG,
    DEFAULT_RAY,
    DEFAULT_SCALE,
    DEFAULT_AUTO_MARGIN_SIGMA,
    DEFAULT_ENERGY_SPREAD_PCT,
    DEFAULT_ENTRANCE_WET_MM,
    DEFAULT_FIELD_EDGE,
    FIELD_EDGES,
    DEFAULT_GAMMA_CUTOFF_PCT,
    DEFAULT_GAMMA_DOSE_PCT,
    DEFAULT_GAMMA_DTA_MM,
    DEFAULT_IC_GAP_MM,
    DEFAULT_MC_HISTORIES,
    DEFAULT_MODEL,
    DEFAULT_PHANTOM_MM,
    DEFAULT_SPOT_CAP,
    DEFAULT_WEIGHT,
    ERROR_ABSOLUTE,
    ERROR_PERCENT,
    GAMMA_ACTION_PCT,
    GAMMA_TOLERANCE_PCT,
    GRAIN_SPOT,
    GRAIN_TIMESLICE,
    MC_HISTORIES,
    MC_MEDIA,
    MEDIUM_A150,
    MEDIUM_ALUMINUM,
    MEDIUM_COPPER,
    MEDIUM_PMMA,
    MEDIUM_POLYETHYLENE,
    MEDIUM_POLYSTYRENE,
    MEDIUM_WATER,
    MODEL_ANALYTIC,
    MODEL_MC,
    DEFAULT_PLAN_SIGMA,
    PLAN_SIGMA_INTERLOCK,
    PLAN_SIGMA_MEASURED,
    PLAN_SIGMA_REFERENCE,
    PRESET_BY_ID,
    PRESETS,
    RAY_INTEGRAL,
    RAY_MAXIMUM,
    RAY_TRANSPARENT,
    WEIGHT_DOSE,
    WEIGHT_MU,
    WEIGHT_PROTONS,
    XY_IC1,
    XY_IC2,
    XY_ISO_RAY,
    XY_PLAN,
    DoseVolumeConfig,
    active_scale,
    scales_for,
)
from .dose_volume_data import (
    build_view_batches,
    load_splat_sessions,
    range_axis_for_medium,
)
from .dose_volume_fill import (
    GAMMA_CMAP,
    MAX_VOXEL_MM,
    MIN_VOXEL_MM,
    VOXEL_MM,
)
from .dose_panes import DEPTH, DVH, GAMMA_HIST, LATERAL, DoseFrame, DoseLayers, DoseWorkspace
from .dose_volume_physics import GammaCriteria
from .dose_volume_raycast import manual_color_limits, suggest_abs_window
from .dose_volume_vispy import DoseScene
from .plot_view_shell import (
    VispyViewWindow,
    make_presets_menu_button,
    make_side_panel_column,
    run_view_window,
)
from .study_source import DELIVERED, LABELS, PLANNED, StudySource
from .vispy_plot import ensure_gl_plus

SOURCE_SESSIONS, SOURCE_STUDY = "sessions", "study"
SESSION_PLOTS, STUDY_PLOTS = (DEPTH, LATERAL), (DVH, GAMMA_HIST)
READBACK_S = 1.0  # slices and plots follow a refining session Monte Carlo at most this often

_GRAIN_ITEMS = (
    (GRAIN_SPOT, "Spot"),
    (GRAIN_TIMESLICE, "Timeslice"),
)
_XY_ITEMS = (
    (XY_IC1, "IC1"),
    (XY_IC2, "IC2"),
    (XY_ISO_RAY, "ISO ray (IC1–IC2)"),
    (XY_PLAN, "Plan"),
)
_MEDIUM_ITEMS = (
    (MEDIUM_WATER, "Water"),
    (MEDIUM_PMMA, "PMMA"),
    (MEDIUM_POLYSTYRENE, "Polystyrene"),
    (MEDIUM_POLYETHYLENE, "Polyethylene"),
    (MEDIUM_A150, "A-150 plastic"),
    (MEDIUM_ALUMINUM, "Aluminum"),
    (MEDIUM_COPPER, "Copper"),
)
_ERROR_ITEMS = (
    (ERROR_PERCENT, "Percent of peak"),
    (ERROR_ABSOLUTE, "Absolute"),
)
_RAY_ITEMS = (
    (RAY_INTEGRAL, "Integrate"),
    (RAY_MAXIMUM, "Maximum"),
    (RAY_TRANSPARENT, "Transparent"),
)
_GAIN_SLIDER_MAX = 100
_SCALE_SLIDER_MAX = 1000
_LABEL_WIDTH = 100


def gamma_pass_rate(tally: tuple[int, int] | None) -> float | None:
    """Percent of scored voxels with γ ≤ 1, or None when nothing was scored."""
    if not tally or tally[1] <= 0:
        return None
    return 100.0 * tally[0] / tally[1]


def gamma_verdict(rate: float | None) -> tuple[str, str]:
    """TG-218 reading of a pass rate and the color to show it in."""
    if rate is None:
        return "No voxels above the cutoff", "#9aa4b2"
    if rate >= GAMMA_TOLERANCE_PCT:
        return "Within tolerance", "#46a758"
    if rate >= GAMMA_ACTION_PCT:
        return "Below tolerance: investigate", "#f5a524"
    return "Below action limit", "#e5484d"


def _log_slider_pos(value: float, lo: float, hi: float) -> int:
    lo_v = max(float(lo), 1e-30)
    hi_v = max(float(hi), lo_v * 1.000001)
    shown = min(max(float(value), lo_v), hi_v)
    span = math.log(hi_v / lo_v)
    t = math.log(shown / lo_v) / span
    return int(round(t * _SCALE_SLIDER_MAX))


def _log_slider_value(pos: int, lo: float, hi: float) -> float:
    lo_v = max(float(lo), 1e-30)
    hi_v = max(float(hi), lo_v * 1.000001)
    t = min(max(int(pos), 0), _SCALE_SLIDER_MAX) / _SCALE_SLIDER_MAX
    return lo_v * (hi_v / lo_v) ** t


class DoseVolumeWindow(VispyViewWindow):
    """Dose from logged sessions in a phantom or from a DICOM study on its CT: slices, 3D volume and plots."""

    def __init__(
        self,
        session_ids: Sequence[str],
        base_dir: str,
        *,
        settings: ViewSettings | None = None,
        initial_preset: str | None = None,
        study: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title="Dose Volume", side_panel_default_width=420, parent=parent)
        self._updating = True
        gl = "gl+" if ensure_gl_plus() else None
        # Built whole before it is shown: moving a live GL widget can drop its context.
        self._workspace = DoseWorkspace(gl=gl, plots=SESSION_PLOTS)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_layout.addWidget(self._workspace, 1)
        self.progress = ProgressLine(self._workspace)
        self._color_axis = self._workspace.volume.color_axis
        self._vispy_canvas = self._workspace.volume.canvas
        self._scene = DoseScene(self._vispy_canvas)
        self._study = StudySource(session_ids, base_dir, criteria=self._gamma_criteria, snapshot=self._snapshot,
                                  parent=self)
        self._readback_at = 0.0
        self._study_share = None
        self._content = None
        self._pushed = None
        self.set_side_panel(self._build_controls())
        # The shared shell caps a side panel at a quarter of the window, which
        # clips this one. 420 is 50 % wider than the usual 280.
        side = self._side_default_width
        total = max(self._splitter.size().width(), self.width(), side + 400)
        self._splitter.setSizes([total - side, side])
        self._update_legend()
        self._scene.range_listener = self._on_gpu_range

        self._session_ids = list(session_ids)
        self._active_session: str | None = None
        self._base_dir = base_dir
        self._sources: dict[str, object] = {}
        self._notes = load_notes(base_dir)
        self._pending_preset = initial_preset
        self._loaded_grain: str | None = None
        self._refresh_generation = 0
        self._residual_active = False
        self._updating = False

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(60)
        self._refresh_timer.timeout.connect(self._start_refresh)
        # Zero-interval ticks run between paint events, so the view stays live while MC refines.
        self._mc_timer = QTimer(self)
        self._mc_timer.setInterval(0)
        self._mc_timer.timeout.connect(self._mc_tick)

        self._load_task = DebouncedBackgroundTask(debounce_ms=0, parent=self)
        self._load_task.finished.connect(self._on_load_finished)
        self._loading = False

        self._study.frameChanged.connect(self._on_study_frame)
        self._study.dosesChanged.connect(self._show_study)
        self._study.analysisChanged.connect(self._show_study)
        self._study.roisChanged.connect(self._on_study_rois)
        self._study.progress.connect(self._on_study_progress)
        self._set_source(SOURCE_STUDY if study or not self._session_ids else SOURCE_SESSIONS)
        if self._session_ids:
            self._start_load()
        if study:
            self._study.load(study)

    def _build_controls(self) -> QWidget:
        panel, outer = make_side_panel_column()
        self._source_combo = SegmentedControl([(SOURCE_SESSIONS, "Sessions"), (SOURCE_STUDY, "DICOM study")])
        self._source_combo.set_button_tooltips({
            SOURCE_SESSIONS: "The selected sessions' logged spots in a water, plastic or metal phantom.",
            SOURCE_STUDY: "A DICOM plan and the selected sessions recalculated on the planning CT.",
        })
        self._source_combo.selectionChanged.connect(self._on_source_changed)
        self._add_row(outer, "Source", self._source_combo)
        # Each source's own controls; what the picture shows and how it is colored follow, shared.
        self._session_box = QWidget()
        layout = QVBoxLayout(self._session_box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        outer.addWidget(self._session_box)
        outer.addWidget(self._study.panel)
        layout.addWidget(
            make_presets_menu_button(
                [(p.id, p.label, True) for p in PRESETS],
                self._apply_preset,
            )
        )

        # Compare is the dose quantity and model. Beam, phantom, and view follow.
        self._session_group = QGroupBox("Session")
        self._session_layout = QVBoxLayout(self._session_group)
        self._session_layout.setSpacing(2)
        self._session_buttons = QButtonGroup(self)
        self._session_buttons.idClicked.connect(self._on_session_picked)
        self._listed_ids: list[str] = []
        self._session_group.setVisible(False)
        layout.addWidget(self._session_group)

        compare_layout = self._add_group(layout, "Compare")
        display_layout = self._add_group(outer, "Display")
        self._show_combo = self._add_segment(
            display_layout, "Show",
            (("dose", "Measured"), ("difference", "− Plan"), ("gamma", "Gamma")),
            self._on_show_changed,
        )
        self._show_combo.set_button_tooltips({
            "dose": "The measured volume.",
            "difference": "Measured minus plan.",
            "gamma": "3D gamma of measured against plan.",
        })
        self._weight_combo = self._add_segment(
            compare_layout, "Quantity",
            ((WEIGHT_DOSE, "Dose"), (WEIGHT_MU, "MU"), (WEIGHT_PROTONS, "Protons")),
            self._on_weight_mode_changed,
        )
        self._weight_combo.set_button_tooltips({
            WEIGHT_DOSE: "Dose in Gy along the Bragg curve.",
            WEIGHT_MU: "Where the logged monitor units stop.",
            WEIGHT_PROTONS: "Where the protons stop.",
        })
        self._weight_combo.set_current(DEFAULT_WEIGHT)
        self._model_combo = self._add_segment(
            compare_layout, "Model",
            ((MODEL_ANALYTIC, "Analytic"), (MODEL_MC, "Monte Carlo")),
            self._on_model_changed,
        )
        self._model_combo.set_button_tooltips({
            MODEL_ANALYTIC: "Fast pencil-beam approximation.",
            MODEL_MC: "GPU Monte Carlo ported from MCsquare. Fills in progressively while the view stays live.",
        })
        self._model_combo.set_current(DEFAULT_MODEL)
        self._model_row = self._model_combo.parentWidget()
        self._histories_combo = self._add_segment(
            compare_layout, "Histories",
            tuple((str(h), f"{h / 1e6:g}M") for h in MC_HISTORIES),
            self._on_controls_changed,
        )
        self._histories_combo.setToolTip("Protons simulated for each volume. 4× the histories halves the noise.")
        self._set_combo(self._histories_combo, str(DEFAULT_MC_HISTORIES))
        self._histories_row = self._histories_combo.parentWidget()
        self._gap_spin = self._add_spin(
            compare_layout, "IC gap", 0.1, 100.0, 0.5, DEFAULT_IC_GAP_MM,
            decimals=1,
        )
        self._gap_spin.setSuffix(" mm")
        self._gap_spin.setToolTip("Chamber gap used to turn logged charge into protons.")
        self._gap_row = self._gap_spin.parentWidget()
        self._gap_row.setVisible(DEFAULT_WEIGHT != WEIGHT_MU)
        self._gamma_group = QGroupBox("Gamma")
        gamma_layout = QVBoxLayout(self._gamma_group)
        gamma_layout.setContentsMargins(8, 12, 8, 8)
        gamma_layout.setSpacing(6)
        self._gamma_dd_spin = self._add_spin(
            gamma_layout, "Dose diff.", 0.5, 20.0, 0.5, DEFAULT_GAMMA_DOSE_PCT, decimals=1,
        )
        self._gamma_dd_spin.setSuffix(" %")
        self._gamma_dd_spin.setToolTip("Global: percent of the plan's maximum dose.")
        self._gamma_dta_spin = self._add_spin(
            gamma_layout, "DTA", 0.5, 10.0, 0.5, DEFAULT_GAMMA_DTA_MM, decimals=1,
        )
        self._gamma_dta_spin.setSuffix(" mm")
        self._gamma_dta_spin.setToolTip("Distance to agreement, searched in 3D.")
        self._gamma_cut_spin = self._add_spin(
            gamma_layout, "Low cutoff", 0.0, 50.0, 1.0, DEFAULT_GAMMA_CUTOFF_PCT, decimals=0,
        )
        self._gamma_cut_spin.setSuffix(" %")
        self._gamma_cut_spin.setToolTip("Skip measured voxels below this share of the plan maximum.")
        self._gamma_label = QLabel("—")
        self._gamma_label.setWordWrap(True)
        self._gamma_label.setToolTip(
            f"AAPM TG-218: ≥ {GAMMA_TOLERANCE_PCT:g} % passing is within tolerance; "
            f"below {GAMMA_ACTION_PCT:g} % calls for action."
        )
        gamma_layout.addWidget(self._gamma_label)
        self._gamma_group.setVisible(False)
        outer.addWidget(self._gamma_group)

        beam_layout = self._add_group(layout, "Beam")
        self._grain_combo = self._add_segment(
            beam_layout, "Data", _GRAIN_ITEMS, self._on_grain_changed,
        )
        self._grain_combo.set_button_tooltips({
            GRAIN_SPOT: "One row per spot.",
            GRAIN_TIMESLICE: "One row per timeslice.",
        })
        self._xy_combo = self._add_combo(
            beam_layout, "Position", _XY_ITEMS, self._on_controls_changed,
        )
        self._xy_combo.setToolTip(
            "IC1 and IC2 are the logged spot positions at isocenter. "
            "ISO ray reconstructs isocenter from the two chambers."
        )
        self._smear_spin = self._add_spin(
            beam_layout, "Energy spread", 0.0, 10.0, 0.1, DEFAULT_ENERGY_SPREAD_PCT,
            decimals=2,
        )
        self._smear_spin.setSuffix(" %")
        self._smear_spin.setToolTip(
            "Beam energy spread σE, percent of energy. Range straggle is added on top."
        )
        self._plan_sigma_combo = self._add_segment(
            beam_layout, "Plan σ",
            (
                (PLAN_SIGMA_MEASURED, "Layer"),
                (PLAN_SIGMA_REFERENCE, "Session"),
                (PLAN_SIGMA_INTERLOCK, "Interlock"),
            ),
            self._on_plan_sigma_changed,
        )
        self._plan_sigma_combo.set_button_tooltips({
            PLAN_SIGMA_MEASURED: "This session's measured spot σ, one value per energy layer.",
            PLAN_SIGMA_REFERENCE: "Measured per-layer σ from another loaded session.",
            PLAN_SIGMA_INTERLOCK: "The interlock σ from devices.xml, in chamber mm.",
        })
        self._plan_sigma_combo.set_current(DEFAULT_PLAN_SIGMA)
        self._ref_combo = QComboBox()
        self._ref_combo.setToolTip("Loaded session whose per-layer σ the plan uses.")
        self._ref_combo.currentIndexChanged.connect(self._on_controls_changed)
        self._add_row(beam_layout, "Reference", self._ref_combo)
        self._ref_row = self._ref_combo.parentWidget()
        self._ref_row.setVisible(DEFAULT_PLAN_SIGMA == PLAN_SIGMA_REFERENCE)
        self._scatter_check = QCheckBox("In medium")
        self._scatter_check.setChecked(True)
        self._scatter_check.setToolTip(
            "Widen measured and plan alike as the beam scatters. Off keeps the entrance σ."
        )
        self._scatter_check.toggled.connect(self._on_controls_changed)
        self._add_row(beam_layout, "Scatter", self._scatter_check)

        phantom_layout = self._add_group(layout, "Phantom")
        self._medium_combo = self._add_combo(
            phantom_layout, "Medium", _MEDIUM_ITEMS, self._on_controls_changed,
        )
        self._medium_combo.currentIndexChanged.connect(self._sync_model_controls)
        self._sync_model_controls()
        self._phantom_spin = self._add_spin(
            phantom_layout, "Thickness", 0.0, 1000.0, 10.0, DEFAULT_PHANTOM_MM, decimals=0,
        )
        self._phantom_spin.setSuffix(" mm")
        self._phantom_spin.setSpecialValueText("Auto")
        self._phantom_spin.setToolTip(
            "Depth along the beam. Auto holds the whole range. A fixed depth clips the exit."
        )
        self._margin_spin = self._add_spin(
            phantom_layout, "Auto margin", 1.0, 5.0, 0.5, DEFAULT_AUTO_MARGIN_SIGMA, decimals=1,
        )
        self._margin_spin.setSuffix(" σ")
        self._margin_spin.setToolTip("How many range-spread σ Auto adds past the deepest spot. 5σ keeps the tail.")
        self._margin_row = self._margin_spin.parentWidget()
        self._phantom_spin.valueChanged.connect(self._sync_phantom_controls)
        self._wet_spin = self._add_spin(
            phantom_layout, "Entrance WET", 0.0, 300.0, 1.0, DEFAULT_ENTRANCE_WET_MM, decimals=1,
        )
        self._wet_spin.setSuffix(" mm")
        self._wet_spin.setToolTip("Tank wall, buildup, or range shifter in front of the phantom, in water-equivalent mm.")
        self._phantom_box_check = QCheckBox()
        self._phantom_box_check.setChecked(False)
        self._phantom_box_check.setToolTip("Draw the phantom outline.")
        self._phantom_box_check.toggled.connect(self._on_controls_changed)
        self._phantom_note = QLabel("—")
        self._add_check_line(phantom_layout, self._phantom_box_check, self._phantom_note)

        view_layout = self._add_group(layout, "View")
        self._gantry_spin = self._add_spin(
            view_layout, "Gantry", 0.0, 360.0, 5.0, DEFAULT_GANTRY_DEG,
            decimals=1,
        )
        self._gantry_spin.setWrapping(True)
        self._gantry_spin.setSuffix(" °")
        self._ray_combo = self._add_combo(
            display_layout, "Ray", _RAY_ITEMS, self._on_ray_changed,
        )
        self._ray_combo.setToolTip("Integrate sums a ray. Maximum keeps its hottest sample. Transparent fades like fog.")
        self._set_combo(self._ray_combo, DEFAULT_RAY)
        self._voxel_spin = self._add_spin(
            view_layout, "Voxel", MIN_VOXEL_MM, MAX_VOXEL_MM, 0.25, VOXEL_MM,
            decimals=2,
        )
        self._voxel_spin.setSuffix(" mm")
        self._voxel_spin.setToolTip("Voxel edge. Smaller is finer and slower, and may crop a large field.")
        self._interp = SegmentedControl([
            ("nearest", "Nearest"),
            ("linear", "Linear"),
            ("cubic", "Cubic"),
        ])
        self._interp.set_current("linear")
        self._interp.set_button_tooltips({
            "nearest": "Show each voxel as stored.",
            "linear": "Blend the neighboring voxels.",
            "cubic": "Smoother, and can overshoot a little.",
        })
        self._interp.selectionChanged.connect(self._scene.set_interp)
        self._add_row(display_layout, "Sample", self._interp)
        self._cap_spin = QSpinBox()
        self._cap_spin.setRange(1_000, 5_000_000)
        self._cap_spin.setSingleStep(50_000)
        self._cap_spin.setValue(DEFAULT_SPOT_CAP)
        self._cap_spin.setToolTip("Most spots drawn. Longer sessions are thinned evenly.")
        self._cap_spin.valueChanged.connect(self._on_controls_changed)
        self._add_row(view_layout, "Spot cap", self._cap_spin)
        self._grid_label = QLabel("—")
        self._grid_label.setToolTip("Voxels along X, Y, and depth.")
        self._grid_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._add_row(display_layout, "Grid", self._grid_label)
        self._spots_label = QLabel("—")
        self._spots_label.setToolTip("Spots in the volume. A plan count appears while comparing.")
        self._spots_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._add_row(view_layout, "Spots", self._spots_label)

        self._seq_scale = DEFAULT_SCALE
        self._div_scale = DEFAULT_DIVERGENT_SCALE
        color_group = QGroupBox("Color")
        color_layout = QVBoxLayout(color_group)
        color_layout.setContentsMargins(8, 12, 8, 8)
        color_layout.setSpacing(6)
        self._scale_combo = self._add_combo(
            color_layout, "Scale", scales_for(False), self._on_scale_changed,
        )
        self._error_combo = self._add_combo(
            color_layout, "Full scale", _ERROR_ITEMS, self._on_error_mode_changed,
        )
        self._scale_mode_row = self._error_combo.parentWidget()
        self._set_combo(self._error_combo, DEFAULT_ERROR_MODE)
        self._gain = DEFAULT_GAIN
        self._abs_edited = False
        self._gain_box = QWidget()
        gain_row = QHBoxLayout(self._gain_box)
        gain_row.setContentsMargins(0, 0, 0, 0)
        self._window_label = QLabel("Window")
        self._window_label.setFixedWidth(_LABEL_WIDTH)
        gain_row.addWidget(self._window_label)
        self._gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._gain_slider.setRange(0, _GAIN_SLIDER_MAX)
        self._gain_slider.setValue(int(round(DEFAULT_GAIN * _GAIN_SLIDER_MAX)))
        self._gain_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._error_scale_spin = QDoubleSpinBox()
        self._error_scale_spin.setMaximumWidth(140)
        self._error_scale_spin.valueChanged.connect(self._on_error_scale_changed)
        self._configure_error_scale_spin(DEFAULT_ERROR_MODE, DEFAULT_ERROR_SCALE)
        self._gain_label = QLabel(f"{DEFAULT_GAIN:.2f}")
        self._gain_label.setMinimumWidth(56)
        self._gain_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        self._gain_slider.valueChanged.connect(self._on_gain_changed)
        self._auto_check = QCheckBox("Auto")
        self._auto_check.setToolTip("Fit the color window to the rays on screen.")
        gain_row.addWidget(self._gain_slider, stretch=1)
        gain_row.addWidget(self._error_scale_spin)
        gain_row.addWidget(self._gain_label)
        gain_row.addWidget(self._auto_check)
        color_layout.addWidget(self._gain_box)
        self._auto_check.toggled.connect(self._on_auto_changed)
        self._auto_check.setChecked(True)
        outer.addWidget(color_group)

        field_layout = self._add_group(layout, "Field Bounds")
        self._field_combo = self._add_combo(
            field_layout, "Edge",
            [(edge_id, label) for edge_id, label, *_rest in FIELD_EDGES],
            self._on_controls_changed,
        )
        self._set_combo(self._field_combo, DEFAULT_FIELD_EDGE)
        self._field_combo.setToolTip(
            "50% of each slice is the lateral field. 90% of the plan is the planned high dose."
        )
        self._field_box_check = QCheckBox()
        self._field_box_check.setChecked(True)
        self._field_box_check.setToolTip("Draw the gold field outline. The size stays either way.")
        self._field_box_check.toggled.connect(self._on_controls_changed)
        self._field_size = QLabel("—")
        self._field_size.setToolTip("X × Y × depth, in millimetres.")
        self._add_check_line(field_layout, self._field_box_check, self._field_size)
        outer.addStretch(1)
        return panel

    def _add_check_line(self, layout: QVBoxLayout, check: QCheckBox, readout: QLabel) -> None:
        """Checkbox and its readback, aligned with the labeled controls above."""
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        gutter = QWidget()
        gutter.setFixedWidth(_LABEL_WIDTH)
        readout.setWordWrap(True)
        readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        readout.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(gutter)
        row.addWidget(check, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(readout, 1)
        layout.addWidget(host)

    def _add_group(self, layout: QVBoxLayout, title: str) -> QVBoxLayout:
        group = QGroupBox(title)
        inner = QVBoxLayout(group)
        inner.setContentsMargins(8, 12, 8, 8)
        inner.setSpacing(6)
        layout.addWidget(group)
        return inner

    def _sync_model_controls(self, *_args) -> None:
        """Model is for Dose, and Monte Carlo only for media MCsquare has data for."""
        supported = self._medium_combo.currentData() in MC_MEDIA
        if not supported:
            self._model_combo.set_current(MODEL_ANALYTIC)
        self._model_combo.setEnabled(supported)
        self._model_combo.setToolTip("" if supported else "No MCsquare material data")
        dose = self._choice(self._weight_combo) == WEIGHT_DOSE
        self._model_row.setVisible(dose)
        mc = dose and self._choice(self._model_combo) == MODEL_MC
        self._histories_row.setVisible(mc)
        self._scatter_check.setEnabled(not mc)

    def _on_model_changed(self, *_args) -> None:
        if self._updating:
            return
        self._sync_model_controls()
        self._schedule_refresh()

    def _sync_phantom_controls(self, *_args) -> None:
        """The margin only shapes an Auto phantom, so it leaves the panel otherwise."""
        auto = self._phantom_spin.value() <= 0.0
        self._margin_spin.setEnabled(auto)
        self._margin_row.setVisible(auto)

    def _add_row(self, layout: QVBoxLayout, label: str, widget: QWidget) -> None:
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        text = QLabel(label)
        text.setFixedWidth(_LABEL_WIDTH)
        row.addWidget(text)
        row.addWidget(widget, stretch=1)
        layout.addWidget(host)

    def _add_combo(
        self,
        layout: QVBoxLayout,
        label: str,
        items: tuple[tuple[str, str], ...],
        handler,
    ) -> QComboBox:
        combo = QComboBox()
        for value, text in items:
            combo.addItem(text, value)
        combo.currentIndexChanged.connect(handler)
        self._add_row(layout, label, combo)
        return combo

    def _add_segment(self, layout: QVBoxLayout, label: str, items, handler) -> SegmentedControl:
        options = [(key, text) for key, text in items]
        control = SegmentedControl(options)
        if options:
            control.set_current(options[0][0])
        control.selectionChanged.connect(handler)
        self._add_row(layout, label, control)
        return control

    def _choice(self, widget) -> str:
        key = widget.current_key() if isinstance(widget, SegmentedControl) else widget.currentData()
        return key or ""

    def _add_spin(
        self,
        layout: QVBoxLayout,
        label: str,
        lo: float,
        hi: float,
        step: float,
        value: float,
        *,
        decimals: int,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.valueChanged.connect(self._on_controls_changed)
        self._add_row(layout, label, spin)
        return spin

    def _gain_value(self) -> float:
        return float(self._gain)

    def _error_scale_value(self) -> float:
        raw = float(self._error_scale_spin.value())
        if self._error_combo.currentData() == ERROR_PERCENT:
            return raw / 100.0
        return raw

    def _deposit_weight(self) -> str:
        combo = getattr(self, "_weight_combo", None)
        if combo is None:
            return DEFAULT_WEIGHT
        return self._choice(combo) or DEFAULT_WEIGHT

    def _density_unit(self, per_area: bool | None = None) -> str:
        if per_area is None:
            per_area = self._ray_combo.currentData() == RAY_INTEGRAL
        if self._dicom():
            return "Gy(RBE)·mm" if per_area else "Gy(RBE)"
        weight = self._deposit_weight()
        if weight == WEIGHT_DOSE:
            return "Gy·mm" if per_area else "Gy"
        unit = "protons" if weight == WEIGHT_PROTONS else "MU"
        return f"{unit} / mm²" if per_area else f"{unit} / mm³"

    def _abs_suffix(self) -> str:
        return " " + self._density_unit().replace(" ", "")

    def _configure_error_scale_spin(self, mode: str, scale: float) -> None:
        if mode == ERROR_PERCENT:
            self._error_scale_spin.setDecimals(0)
            self._error_scale_spin.setRange(1.0, 100.0)
            self._error_scale_spin.setSingleStep(1.0)
            self._error_scale_spin.setSuffix(" %")
            self._error_scale_spin.setValue(max(1.0, min(100.0, float(scale) * 100.0)))
            return
        suffix = self._abs_suffix()
        shown = max(float(scale), 1e-12)
        if self._deposit_weight() == WEIGHT_PROTONS:
            self._error_scale_spin.setDecimals(3)
            self._error_scale_spin.setRange(1.0, 1.0e12)
            self._error_scale_spin.setSingleStep(max(shown * 0.1, 1.0))
            self._error_scale_spin.setSuffix(suffix)
            self._error_scale_spin.setValue(max(1.0, min(1.0e12, shown)))
            return
        decimals = 3 if shown >= 1.0 else 6 if shown >= 1.0e-4 else 8
        self._error_scale_spin.setDecimals(decimals)
        self._error_scale_spin.setRange(1.0e-8, 1.0e6)
        self._error_scale_spin.setSingleStep(max(shown * 0.1, 1.0e-8))
        self._error_scale_spin.setSuffix(suffix)
        self._error_scale_spin.setValue(max(1.0e-8, min(1.0e6, shown)))

    def _comparing(self) -> bool:
        return self._choice(self._show_combo) == "difference"

    def _gamma_mode(self) -> bool:
        return self._choice(self._show_combo) == "gamma"

    def _gamma_criteria(self) -> GammaCriteria:
        return GammaCriteria(
            dose_pct=self._gamma_dd_spin.value(),
            dta_mm=self._gamma_dta_spin.value(),
            cutoff_pct=self._gamma_cut_spin.value(),
        )

    def _fill_scale_combo(self, compare: bool) -> None:
        chosen = self._div_scale if compare else self._seq_scale
        self._scale_combo.blockSignals(True)
        self._scale_combo.clear()
        for value, text in scales_for(compare):
            self._scale_combo.addItem(text, value)
        self._set_combo(self._scale_combo, active_scale(compare, chosen))
        self._scale_combo.blockSignals(False)

    def _slider_is_window(self) -> bool:
        """Difference, other than transparent: the slider is the full-scale number."""
        return self._comparing() and self._ray_combo.currentData() != RAY_TRANSPARENT

    def _window_slider_pos(self) -> int:
        shown = float(self._error_scale_spin.value())
        lo = float(self._error_scale_spin.minimum())
        hi = float(self._error_scale_spin.maximum())
        if self._error_combo.currentData() == ERROR_PERCENT:
            span = hi - lo
            if span <= 0.0:
                return 0
            t = (min(max(shown, lo), hi) - lo) / span
            return int(round(t * _SCALE_SLIDER_MAX))
        return _log_slider_pos(shown, lo, hi)

    def _window_slider_value(self) -> float:
        lo = float(self._error_scale_spin.minimum())
        hi = float(self._error_scale_spin.maximum())
        pos = int(self._gain_slider.value())
        if self._error_combo.currentData() == ERROR_PERCENT:
            t = min(max(pos, 0), _SCALE_SLIDER_MAX) / float(_SCALE_SLIDER_MAX)
            return lo + t * (hi - lo)
        return _log_slider_value(pos, lo, hi)

    def _place_slider(self) -> None:
        slider = self._gain_slider
        slider.blockSignals(True)
        if self._slider_is_window():
            slider.setRange(0, _SCALE_SLIDER_MAX)
            slider.setValue(self._window_slider_pos())
        else:
            slider.setRange(0, _GAIN_SLIDER_MAX)
            slider.setValue(int(round(max(0.0, min(1.0, self._gain)) * _GAIN_SLIDER_MAX)))
        slider.blockSignals(False)

    def _display_peak(self) -> float:
        integral = self._ray_combo.currentData() == RAY_INTEGRAL
        peak = self._scene.ray_peak if integral else self._scene.dose_peak
        return max(float(peak), 1e-12)

    def _maybe_seed_absolute_window(self) -> None:
        """Point an untouched absolute window at the dose, not a fixed 0.2 MU."""
        if self._abs_edited or not self._comparing():
            return
        if self._error_combo.currentData() != ERROR_ABSOLUTE:
            return
        if not self._scene._has_volume:
            return
        suggested = suggest_abs_window(self._display_peak())
        if abs(self._error_scale_value() - suggested) <= 1e-9 * max(1.0, suggested):
            return
        self._updating = True
        try:
            self._configure_error_scale_spin(ERROR_ABSOLUTE, suggested)
            self._place_slider()
        finally:
            self._updating = False
        self._scene.set_error_metric(ERROR_ABSOLUTE, self._error_scale_value())

    def _sync_color_controls(self) -> None:
        diff = self._comparing()
        transparent = self._ray_combo.currentData() == RAY_TRANSPARENT
        auto = self._auto_check.isChecked()
        linked = self._slider_is_window()
        # γ has fixed limits, its own map, and always shows the worst voxel per ray.
        gamma = self._gamma_mode()
        if hasattr(self, "_gamma_group"):  # the Auto box syncs while the panel is still building
            self._gamma_group.setVisible(gamma)
        self._gain_box.setVisible(not gamma)
        self._scale_combo.setEnabled(not gamma)
        self._ray_combo.setEnabled(not gamma)
        self._scale_mode_row.setVisible(diff)
        self._error_scale_spin.setVisible(diff)
        self._gain_label.setVisible(not linked)
        self._window_label.setText("Opacity" if transparent else "Window")
        self._gain_slider.setEnabled(transparent or not auto)
        self._error_scale_spin.setEnabled((not auto) or transparent)
        if linked:
            self._gain_slider.setToolTip(
                "Full-scale window. The same number as the box beside it."
            )
        elif transparent:
            self._gain_slider.setToolTip("Opacity of the transparent rays.")
        else:
            self._gain_slider.setToolTip(
                "Color window. Auto fits it to the brightest ray on screen."
            )
        self._place_slider()

    def _read_config(self) -> DoseVolumeConfig:
        compare = self._comparing()
        gamma = self._gamma_mode()
        crit = self._gamma_criteria()
        return DoseVolumeConfig(
            grain=self._choice(self._grain_combo),
            xy_mode=self._xy_combo.currentData(),
            overlay_plan=compare or gamma,
            plan_sigma=self._choice(self._plan_sigma_combo),
            plan_sigma_ref=self._ref_combo.currentData() or "",
            scatter=self._scatter_check.isChecked(),
            show_phantom=self._phantom_box_check.isChecked(),
            show_field=self._field_box_check.isChecked(),
            gamma=gamma,
            gamma_dose_pct=crit.dose_pct,
            gamma_dta_mm=crit.dta_mm,
            gamma_cutoff_pct=crit.cutoff_pct,
            error_mode=self._error_combo.currentData(),
            error_scale=self._error_scale_value(),
            weight_mode=self._choice(self._weight_combo),
            dose_model=self._choice(self._model_combo) or DEFAULT_MODEL,
            mc_histories=int(self._choice(self._histories_combo) or DEFAULT_MC_HISTORIES),
            ic_gap_mm=self._gap_spin.value(),
            gain=self._gain_value(),
            smear_axis_units=self._smear_spin.value(),
            splat_cap=self._cap_spin.value(),
            gantry_deg=self._gantry_spin.value(),
            medium=self._medium_combo.currentData(),
            phantom_mm=self._phantom_spin.value(),
            auto_margin_sigma=self._margin_spin.value(),
            entrance_wet_mm=self._wet_spin.value(),
            field_edge=self._field_combo.currentData() or DEFAULT_FIELD_EDGE,
            ray_mode=self._ray_combo.currentData(),
            scale=active_scale(compare, self._scale_combo.currentData() or DEFAULT_SCALE),
            auto_scale=self._auto_check.isChecked(),
            voxel_mm=self._voxel_spin.value(),
            interp=self._interp.current_key() or "linear",
        )

    def _set_combo(self, combo, value: str) -> None:
        if isinstance(combo, SegmentedControl):
            combo.set_current(value)
            return
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _set_config(self, config: DoseVolumeConfig) -> None:
        self._updating = True
        try:
            self._set_combo(self._grain_combo, config.grain)
            self._set_combo(self._xy_combo, config.xy_mode)
            self._set_combo(self._plan_sigma_combo, config.plan_sigma)
            self._ref_row.setVisible(config.plan_sigma == PLAN_SIGMA_REFERENCE)
            self._set_combo(self._ref_combo, config.plan_sigma_ref)
            self._scatter_check.setChecked(config.scatter)
            self._phantom_box_check.setChecked(config.show_phantom)
            self._field_box_check.setChecked(config.show_field)
            show = "gamma" if config.gamma else "difference" if config.overlay_plan else "dose"
            self._set_combo(self._show_combo, show)
            compare = show == "difference"
            allowed = {name for name, _label in scales_for(compare)}
            if config.scale in allowed:
                if compare:
                    self._div_scale = config.scale
                else:
                    self._seq_scale = config.scale
            self._fill_scale_combo(compare)
            self._gamma_dd_spin.setValue(config.gamma_dose_pct)
            self._gamma_dta_spin.setValue(config.gamma_dta_mm)
            self._gamma_cut_spin.setValue(config.gamma_cutoff_pct)
            self._set_combo(self._error_combo, config.error_mode)
            self._configure_error_scale_spin(config.error_mode, config.error_scale)
            self._set_combo(self._medium_combo, config.medium)
            self._phantom_spin.setValue(config.phantom_mm)
            self._margin_spin.setValue(config.auto_margin_sigma)
            self._sync_phantom_controls()
            self._wet_spin.setValue(config.entrance_wet_mm)
            self._set_combo(self._field_combo, config.field_edge)
            self._set_combo(self._weight_combo, config.weight_mode)
            self._set_combo(self._model_combo, config.dose_model)
            self._set_combo(self._histories_combo, str(config.mc_histories))
            self._sync_model_controls()
            self._set_combo(self._ray_combo, config.ray_mode)
            self._auto_check.setChecked(config.auto_scale)
            self._gain = max(0.0, min(1.0, float(config.gain)))
            self._sync_color_controls()
            self._gap_spin.setValue(config.ic_gap_mm)
            self._gap_spin.setEnabled(config.weight_mode != WEIGHT_MU)
            self._smear_spin.setValue(config.smear_axis_units)
            self._cap_spin.setValue(config.splat_cap)
            self._voxel_spin.setValue(config.voxel_mm)
            self._interp.set_current(config.interp)
            self._gantry_spin.setValue(config.gantry_deg)
        finally:
            self._updating = False

    def _apply_preset(self, preset_id: str) -> None:
        preset = PRESET_BY_ID.get(preset_id)
        if preset is None:
            return
        config = self._read_config()
        config.grain = preset.grain
        config.xy_mode = preset.xy_mode
        config.overlay_plan = preset.overlay_plan
        config.gamma = False
        grain_changed = config.grain != self._loaded_grain
        self._set_config(config)
        if grain_changed:
            self._start_load()
        else:
            self._schedule_refresh()

    def _on_grain_changed(self, *_args) -> None:
        if self._updating:
            return
        self._start_load()

    def _on_show_changed(self, *_args) -> None:
        if self._updating:
            return
        self._fill_scale_combo(self._comparing())
        self._sync_color_controls()
        self._maybe_seed_absolute_window()
        self._schedule_refresh()

    def _on_scale_changed(self, *_args) -> None:
        if self._updating:
            return
        name = self._scale_combo.currentData()
        if not name:
            return
        if self._comparing():
            self._div_scale = name
        else:
            self._seq_scale = name
        self._scene.set_scale(name)
        self._update_legend()

    def _on_error_mode_changed(self, *_args) -> None:
        if self._updating:
            return
        mode = self._error_combo.currentData()
        self._abs_edited = False
        default = DEFAULT_ERROR_PCT / 100.0 if mode == ERROR_PERCENT else DEFAULT_ERROR_MU
        self._updating = True
        try:
            self._configure_error_scale_spin(mode, default)
        finally:
            self._updating = False
        self._maybe_seed_absolute_window()
        self._sync_color_controls()
        self._apply_error_metric()

    def _on_error_scale_changed(self, *_args) -> None:
        if self._updating:
            return
        if self._error_combo.currentData() == ERROR_ABSOLUTE:
            self._abs_edited = True
        if self._slider_is_window():
            self._place_slider()
        self._apply_error_metric()

    def _apply_error_metric(self) -> None:
        mode = self._error_combo.currentData()
        scale = self._error_scale_value()
        self._scene.set_error_metric(mode, scale)
        self._update_legend()

    def _on_weight_mode_changed(self, *_args) -> None:
        if self._updating:
            return
        show_gap = self._choice(self._weight_combo) != WEIGHT_MU
        self._gap_spin.setEnabled(show_gap)
        self._gap_row.setVisible(show_gap)
        self._sync_model_controls()
        # New units: an absolute window typed for the old ones means nothing now.
        self._abs_edited = False
        if self._error_combo.currentData() == ERROR_ABSOLUTE:
            self._updating = True
            try:
                self._error_scale_spin.setSuffix(self._abs_suffix())
            finally:
                self._updating = False
        self._schedule_refresh()

    def _on_ray_changed(self, *_args) -> None:
        if self._updating:
            return
        self._scene.set_ray(self._ray_combo.currentData())
        self._abs_edited = False
        if self._error_combo.currentData() == ERROR_ABSOLUTE:
            self._updating = True
            try:
                self._error_scale_spin.setSuffix(self._abs_suffix())
            finally:
                self._updating = False
        self._maybe_seed_absolute_window()
        self._sync_color_controls()
        self._refresh_window_readout()
        self._update_legend()

    def _on_auto_changed(self, *_args) -> None:
        self._sync_color_controls()
        self._refresh_window_readout()
        if self._updating:
            return
        self._scene.set_auto(self._auto_check.isChecked())
        self._update_legend()

    def _on_gpu_range(self) -> None:
        self._refresh_window_readout()
        self._update_legend()

    def _refresh_window_readout(self) -> None:
        if self._slider_is_window():
            return
        if self._ray_combo.currentData() == RAY_TRANSPARENT:
            self._gain_label.setText(f"{self._gain:.2f}")
            return
        _lo, hi = self._legend_numbers()
        self._gain_label.setText(f"{hi:.3g}")

    def _on_gain_changed(self, *_args) -> None:
        if self._updating:
            return
        if self._slider_is_window():
            value = self._window_slider_value()
            self._updating = True
            try:
                self._error_scale_spin.setValue(value)
            finally:
                self._updating = False
            if self._error_combo.currentData() == ERROR_ABSOLUTE:
                self._abs_edited = True
            self._apply_error_metric()
            return
        self._gain = self._gain_slider.value() / float(_GAIN_SLIDER_MAX)
        self._refresh_window_readout()
        self._scene.set_gain(self._gain)
        self._update_legend()

    def _drawn_difference(self) -> bool:
        """What the volume actually shows; Difference falls back to dose without a plan."""
        return self._comparing() and self._scene.difference

    def _legend_numbers(self) -> tuple[float, float]:
        if self._scene.gamma:
            return 0.0, float(self._gamma_criteria().cap)
        auto = self._auto_check.isChecked()
        percent = (
            self._drawn_difference()
            and self._error_combo.currentData() == ERROR_PERCENT
        )
        if auto and self._scene.auto_lo is not None and self._scene.auto_hi is not None:
            lo, hi = float(self._scene.auto_lo), float(self._scene.auto_hi)
            if percent:
                peak = self._display_peak()
                return 100.0 * lo / peak, 100.0 * hi / peak
            return lo, hi
        if percent:
            pct = self._error_scale_value() * 100.0
            return -pct, pct
        mode = self._ray_combo.currentData()
        integral = mode == RAY_INTEGRAL
        return manual_color_limits(
            difference=self._drawn_difference(),
            transparent=mode == RAY_TRANSPARENT,
            integral=integral,
            gain=self._gain_value(),
            typical=float(self._scene.dose_peak),
            ray_scale=float(self._scene.ray_peak) if integral else float(self._scene.dose_peak),
            error_scale=self._error_scale_value(),
            absolute=self._error_combo.currentData() == ERROR_ABSOLUTE,
        )

    def _legend_title(self, *, percent: bool) -> str:
        if self._scene.gamma:
            crit = self._gamma_criteria()
            title = f"γ {crit.dose_pct:g} % / {crit.dta_mm:g} mm"
            rate = self._gamma_rate()
            return title if rate is None else f"{title} · {rate:.1f} % pass"
        density = self._density_unit()
        if self._drawn_difference():
            what = "deliv − plan" if self._dicom() else "meas − plan"
            if percent:
                return f"{what} (% of peak)"
            return f"{what} ({density})"
        return density

    def _update_legend(self) -> None:
        compare = self._drawn_difference()
        name = active_scale(compare, self._scale_combo.currentData() or DEFAULT_SCALE)
        if self._scene.gamma:
            name = GAMMA_CMAP
        percent = compare and self._error_combo.currentData() == ERROR_PERCENT
        lo, hi = self._legend_numbers()
        self._color_axis.set_scale(name, lo, hi, self._legend_title(percent=percent))
        self._push_layers()

    def _on_controls_changed(self, *_args) -> None:
        if self._updating:
            return
        self._schedule_refresh()

    def _on_plan_sigma_changed(self, *_args) -> None:
        self._ref_row.setVisible(self._choice(self._plan_sigma_combo) == PLAN_SIGMA_REFERENCE)
        self._on_controls_changed()

    def _gamma_rate(self) -> float | None:
        if self._dicom():
            g = self._study.gamma
            return None if g is None else 100.0 * g.rate
        return gamma_pass_rate(self._scene.gamma_pass)

    def _show_gamma_verdict(self) -> None:
        pending = not self._study.done if self._dicom() else self._scene.gamma_pending
        if pending and self._gamma_mode():
            self._gamma_label.setText("Waiting for the Monte Carlo to finish")
            self._gamma_label.setStyleSheet("")
            return
        if not self._scene.gamma:
            self._gamma_label.setText(self._no_plan_reason() if self._gamma_mode() else "—")
            self._gamma_label.setStyleSheet("")
            return
        rate = self._gamma_rate()
        verdict, color = gamma_verdict(rate)
        if self._dicom():
            head = "—" if rate is None else f"{rate:.1f} % pass ({self._study.gamma.evaluated:,} voxels) vs TPS"
        else:
            tally = self._scene.gamma_pass
            head = "—" if rate is None else f"{rate:.1f} % pass ({tally[0]:,} of {tally[1]:,} voxels)"
        self._gamma_label.setText(f"{head}\n{verdict}")
        self._gamma_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _update_session_list(self, loaded_ids: list[str]) -> None:
        """One radio per loaded session; rebuilt only when the set changes, so the pick holds."""
        if self._active_session not in loaded_ids:
            self._active_session = loaded_ids[0] if loaded_ids else None
        if loaded_ids == self._listed_ids:
            return
        self._listed_ids = list(loaded_ids)
        for button in self._session_buttons.buttons():
            self._session_buttons.removeButton(button)
            button.deleteLater()
        for i, sid in enumerate(loaded_ids):
            text = format_session_legend_label(sid, self._notes)
            radio = QRadioButton(text)
            radio.setToolTip(text)
            radio.setChecked(sid == self._active_session)
            self._session_buttons.addButton(radio, i)
            self._session_layout.addWidget(radio)
        self._session_group.setVisible(len(loaded_ids) > 1)
        kept = self._ref_combo.currentData()
        others = [sid for sid in loaded_ids if sid != self._active_session]
        self._ref_combo.blockSignals(True)
        self._ref_combo.clear()
        for sid in loaded_ids:
            self._ref_combo.addItem(format_session_legend_label(sid, self._notes), sid)
        if loaded_ids:
            self._set_combo(self._ref_combo, kept if kept in loaded_ids else (others or loaded_ids)[0])
        self._ref_combo.blockSignals(False)

    def _on_session_picked(self, index: int) -> None:
        if 0 <= index < len(self._listed_ids) and self._listed_ids[index] != self._active_session:
            self._active_session = self._listed_ids[index]
            self._scene.reframe()
            self._schedule_refresh()

    def _show_status(self, message: str) -> None:
        if self._dicom():
            return
        self._workspace.clear("")
        self._content = None
        self._scene.render(
            None, None, range_axis_for_medium(MEDIUM_WATER), gain=1.0, status=message,
        )

    def _start_load(self) -> None:
        config = self._read_config()
        session_ids = list(self._session_ids)
        base_dir = self._base_dir
        grain = config.grain

        def loader() -> dict:
            return load_splat_sessions(session_ids, base_dir, grain)

        self._show_status("Loading dose data…")
        self._loading = True
        self._sync_progress()
        self._load_task.schedule(loader)

    @Slot(int, object)
    def _on_load_finished(self, gen: int, result: object) -> None:
        if gen != self._load_task.generation:
            return
        self._loading = False
        self._sync_progress()
        if not isinstance(result, dict):
            self._show_status("Failed to load dose data")
            return
        self._sources = result
        self._loaded_grain = self._read_config().grain
        if not self._sources:
            self._show_status("No IC position / sigma data found")
            return
        preset = self._pending_preset
        self._pending_preset = None
        if preset and preset in PRESET_BY_ID:
            self._apply_preset(preset)
        else:
            self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        self._refresh_generation += 1
        self._refresh_timer.start()
        self._sync_progress()

    def _sync_progress(self) -> None:
        """Busy while loading or about to refresh; the Monte Carlo's share while it refines."""
        if self._dicom():
            share = self._study_share
            if share is None:
                self.progress.done()
            elif share < 0:
                self.progress.busy()
            else:
                self.progress.set_progress(share)
            return
        mc = self._scene.mc_progress
        if self._loading or (mc is None and self._refresh_timer.isActive()):
            self.progress.busy()
        elif mc is not None:
            self.progress.set_progress(mc)
        else:
            self.progress.done()

    def _start_refresh(self) -> None:
        if self._dicom():
            g = self._study.gamma
            if self._study.done and (g is None or g.criteria != self._gamma_criteria()):
                self._study.update_gamma()  # redraws through analysisChanged
            else:
                self._show_study()
            self._sync_progress()
            return
        gen = self._refresh_generation
        self._update_session_list([sid for sid in self._session_ids if sid in self._sources])
        config = self._read_config()
        if gen != self._refresh_generation:
            return
        self.setWindowTitle(config.title)
        measured_batch, plan_batch, axis, _n_raw = build_view_batches(
            self._sources,
            [self._active_session] if self._active_session else [],
            config,
            self._base_dir,
        )

        self._smear_spin.setSuffix(f" {axis.smear_label}")
        has_dose = (
            measured_batch is not None
            and measured_batch.dose_mu is not None
            and bool(np.isfinite(measured_batch.dose_mu).any())
        )
        residual = bool(
            config.overlay_plan
            and plan_batch is not None
            and plan_batch.center.size
            and measured_batch is not None
            and measured_batch.center.size
        )
        self._residual_active = residual
        if gen != self._refresh_generation:
            return
        self._scene.render(
            measured_batch, plan_batch, axis,
            gain=config.gain, gantry_deg=config.gantry_deg,
            difference=residual,
            error_mode=config.error_mode, error_scale=config.error_scale,
            weight_mode=config.weight_mode, ic_gap_mm=config.ic_gap_mm,
            smear=config.smear_axis_units, ray_mode=config.ray_mode,
            scale=config.scale, auto_scale=config.auto_scale,
            voxel_mm=config.voxel_mm, interp=config.interp,
            gamma=config.gamma and residual, gamma_criteria=self._gamma_criteria(),
            phantom_mm=config.phantom_mm, entrance_wet_mm=config.entrance_wet_mm,
            auto_margin_sigma=config.auto_margin_sigma,
            scatter=config.scatter,
            field_edge=config.field_edge,
            show_phantom=config.show_phantom,
            show_field=config.show_field,
            model=config.dose_model,
            mc_histories=config.mc_histories,
        )
        self._show_scene_readouts()
        self._push_session_layers()
        if self._scene.mc_refining:
            self._mc_timer.start()
        else:
            self._mc_timer.stop()
        self._sync_progress()
        n_meas = 0 if measured_batch is None else int(measured_batch.center.shape[0])
        spots = f"{n_meas:,}"
        if residual:
            spots += f" · plan {int(plan_batch.center.shape[0]):,}"
        if config.overlay_plan and not residual:
            spots += " · no plan"
        if measured_batch is not None and not has_dose:
            spots += " · relative" if config.grain == GRAIN_TIMESLICE else " · plan MU"
        self._spots_label.setText(spots)

    def _show_scene_readouts(self) -> None:
        self._show_field_extent()
        self._phantom_note.setText(self._scene.phantom_note or "—")
        self._grid_label.setText(self._scene.volume_note or "—")
        self._maybe_seed_absolute_window()
        self._update_legend()
        self._show_gamma_verdict()

    def _mc_tick(self) -> None:
        refining = self._scene.mc_step()
        self._sync_progress()
        if refining:
            self._grid_label.setText(self._scene.volume_note or "—")
            if time.perf_counter() - self._readback_at > READBACK_S:
                self._push_session_layers()
            return
        self._mc_timer.stop()
        self._show_scene_readouts()
        self._push_session_layers()

    def _show_field_extent(self) -> None:
        ext = getattr(self._scene, "field_extent", None)
        self._field_size.setText("—" if ext is None else f"{ext[0]:.0f} × {ext[1]:.0f} × {ext[2]:.0f} mm")

    def _no_plan_reason(self) -> str:
        if self._dicom():
            if self._study.main_kind != DELIVERED:
                return "Difference is delivered minus plan: pick the Delivered dose"
            return "Gamma is against the TPS dose once the Monte Carlo finishes"
        if self._xy_combo.currentData() == XY_PLAN:
            return "XY is Plan, so there is no separate plan to compare"
        return "No plan to compare against"

    # ---- sources --------------------------------------------------------------------------------

    def _dicom(self) -> bool:
        return self._choice(self._source_combo) == SOURCE_STUDY

    def _set_source(self, key: str) -> None:
        self._source_combo.set_current(key)
        self._on_source_changed()

    def _on_source_changed(self, *_args) -> None:
        dicom = self._dicom()
        self._session_box.setVisible(not dicom)
        self._study.panel.setVisible(dicom)
        self._show_combo.set_option_text("dose", "Dose" if dicom else "Measured")
        self._show_combo.set_button_tooltips({
            "dose": "The dose picked in Study." if dicom else "The measured volume.",
            "difference": "Delivered minus plan." if dicom else "Measured minus plan.",
            "gamma": "Gamma against the TPS dose." if dicom else "3D gamma of measured against plan.",
        })
        self._workspace.set_plot_kinds(STUDY_PLOTS if dicom else SESSION_PLOTS)
        self._content = None
        self._mc_timer.stop()
        if dicom:
            self._scene._mc_stop()
            self._on_study_frame()
        else:
            self._workspace.clear("")
            if self._sources:
                self._schedule_refresh()
            elif self._session_ids:
                self._show_status("Loading dose data…" if self._loading else "No IC position / sigma data found")
            else:
                self._show_status("No sessions selected")
        self._sync_progress()

    def _on_study_frame(self) -> None:
        if not self._dicom():
            return
        frame = self._study.frame()
        if frame is None:
            self._workspace.clear("Open a DICOM study folder")
            self._scene.show_status("Open a DICOM study folder")
            return
        self._workspace.set_frame(frame, self._study.iso_index)
        self._workspace.set_checked(self._study.checked())
        self._show_study()

    def _on_study_rois(self) -> None:
        if self._dicom():
            self._workspace.set_checked(self._study.checked())

    def _on_study_progress(self, share) -> None:
        self._study_share = share
        self._sync_progress()

    def _show_study(self) -> None:
        """The study's doses into the 3D view, slices and plots, as Show asks."""
        s = self._study
        frame = s.frame()
        if not self._dicom() or frame is None:
            return
        main = s.main_kind
        dose, plan = s.doses.get(main), s.doses.get(PLANNED)
        if dose is None:
            return
        show = self._choice(self._show_combo)
        difference = show == "difference" and main == DELIVERED and plan is not None
        gamma = s.gamma_grid if show == "gamma" else None
        self.setWindowTitle(f"Dose Volume: {s.title}")
        self._scene.show_volumes(
            frame.origin, frame.spacing, dose, plan if difference else None, gamma,
            names=frame.names, difference=difference, gamma_cap=self._gamma_criteria().cap,
            beam_dir=frame.beam_dir, note="{} × {} × {} at {:g} × {:g} × {:g} mm".format(*frame.shape, *frame.spacing),
        )
        g = s.gamma
        self._set_content(
            wash=gamma if gamma is not None else dose - plan if difference else dose,
            gamma=gamma is not None, difference=difference,
            profiles={LABELS[k]: v for k, v in s.doses.items()}, main=LABELS[main],
            dvh={LABELS[k]: v for k, v in s.dvh.items()},
            gamma_values=None if g is None else g.gamma,
            gamma_note="" if g is None else
            f"γ {g.criteria.dose_pct:g}%/{g.criteria.dta_mm:g}mm · {100 * g.rate:.1f} % pass",
        )
        self._show_scene_readouts()

    def _push_session_layers(self) -> None:
        """Read the session volume back from the GPU for the slices and plots."""
        self._readback_at = time.perf_counter()
        f = self._scene.frame
        vols = self._scene.volumes() if f is not None else {}
        if not vols:
            self._workspace.clear(self._scene.volume_note or "")
            self._content = None
            return
        origin, spacing, shape, names, beam_dir = f
        frame = DoseFrame(origin=origin, spacing=spacing, shape=tuple(shape), names=names, beam_dir=beam_dir)
        old = self._workspace.frame
        if old is None or old.shape != frame.shape or not np.allclose(old.origin, origin):
            self._workspace.set_frame(frame)
        first, second = vols["first"], vols.get("second")
        difference = self._drawn_difference() and second is not None
        gamma = vols.get("gamma")
        profiles = {"Measured": first} if second is None else {"Measured": first, "Plan": second}
        self._set_content(
            wash=gamma if gamma is not None else first - second if difference else first,
            gamma=gamma is not None, difference=difference, profiles=profiles, main="Measured",
            gamma_values=gamma,
            gamma_note="" if gamma is None else self._gamma_label.text().split("\n")[0],
        )

    def _set_content(self, *, difference: bool, **content) -> None:
        self._content = (difference, content)
        self._pushed = None
        self._push_layers()

    def _push_layers(self) -> None:
        """Slices take the 3D view's colors; only a new scale or window redraws them."""
        if getattr(self, "_content", None) is None or self._workspace.frame is None:
            return
        difference, c = self._content
        name = GAMMA_CMAP if c["gamma"] else active_scale(difference, self._scale_combo.currentData() or DEFAULT_SCALE)
        lo, hi = self._slice_limits(c["wash"], c["gamma"], difference)
        if (name, lo, hi) == self._pushed:
            return
        self._pushed = (name, lo, hi)
        self._workspace.set_layers(DoseLayers(scale=name, lo=lo, hi=hi, unit=self._density_unit(per_area=False), **c))

    def _slice_limits(self, wash, gamma: bool, difference: bool) -> tuple[float, float]:
        """The color axis's numbers, unless they are per ray (Integrate) or a percent, which a slice is not."""
        if gamma:
            return 0.0, float(self._gamma_criteria().cap)
        percent = difference and self._error_combo.currentData() == ERROR_PERCENT
        if self._ray_combo.currentData() != RAY_INTEGRAL and not percent:
            return self._legend_numbers()
        peak = float(np.abs(wash).max()) if wash is not None and wash.size else 0.0
        if percent:
            peak = self._error_scale_value() * float(self._scene.dose_peak)
        peak = peak or 1.0
        return (-peak, peak) if difference else (0.0, peak)

    def _snapshot(self) -> bytes:
        """The whole workspace as PNG bytes, for the report."""
        from PySide6.QtCore import QBuffer, QIODevice, QPoint, QRect
        from PySide6.QtGui import QImage, QPainter

        ws, native = self._workspace, self._vispy_canvas.native
        pix = ws.grab()  # misses the 3D canvas, which is painted in from the scene
        if native.isVisible():  # a never-shown canvas has no GL context
            img = np.ascontiguousarray(self._scene.snapshot())
            h, w = img.shape[:2]
            painter = QPainter(pix)
            painter.drawImage(QRect(native.mapTo(ws, QPoint(0, 0)), native.size()),
                              QImage(img.data, w, h, 4 * w, QImage.Format.Format_RGBA8888))
            painter.end()
        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        pix.save(buf, "PNG")
        return bytes(buf.data())

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt
        self._study.close()
        super().closeEvent(event)


def run_dose_volume_window(
    session_ids: Sequence[str],
    base_dir: str = "test_data",
    *,
    settings: ViewSettings | None = None,
    initial_preset: str | None = None,
    study: str | None = None,
) -> None:
    run_view_window(
        lambda: DoseVolumeWindow(
            session_ids,
            base_dir,
            settings=settings,
            initial_preset=initial_preset,
            study=study,
        ),
        maximize=True,
    )
