"""Qt shell for the 3D dose-volume viewer (visPy)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen
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
    DEFAULT_PHANTOM_MM,
    DEFAULT_SPOT_CAP,
    DEFAULT_WEIGHT,
    ERROR_ABSOLUTE,
    ERROR_PERCENT,
    GAMMA_ACTION_PCT,
    GAMMA_TOLERANCE_PCT,
    GRAIN_SPOT,
    GRAIN_TIMESLICE,
    MEDIUM_A150,
    MEDIUM_ALUMINUM,
    MEDIUM_COPPER,
    MEDIUM_PMMA,
    MEDIUM_POLYETHYLENE,
    MEDIUM_POLYSTYRENE,
    MEDIUM_WATER,
    DEFAULT_PLAN_SIGMA,
    DEFAULT_SIGMA_PLANE,
    PLAN_SIGMA_INTERLOCK,
    PLAN_SIGMA_MEASURED,
    PLAN_SIGMA_REFERENCE,
    PRESET_BY_ID,
    PRESETS,
    RAY_INTEGRAL,
    RAY_MAXIMUM,
    RAY_TRANSPARENT,
    SIGMA_PLANE_CHAMBER,
    SIGMA_PLANE_ISO,
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
    colormap_samples,
    ink_rgb,
    zero_rgb,
)
from .dose_volume_physics import GammaCriteria
from .dose_volume_raycast import manual_color_limits, suggest_abs_window
from .dose_volume_vispy import DoseScene
from .plot_view_shell import (
    VispyViewWindow,
    make_presets_menu_button,
    make_side_panel_column,
    run_view_window,
)
from .vispy_plot import BG, ensure_gl_plus

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


_AXIS_PAD_X = 8
_AXIS_BAR_W = 16
# Fixed so tick text never shoves the view sideways; the ×10 offset rides in the title.
_AXIS_WIDTH = 124
_EXP_DIGITS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

def _tick_labels(values: np.ndarray) -> tuple[list[str], str]:
    """Short tick text plus a shared power-of-ten offset when the span is tiny or huge."""
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return [], ""
    peak = float(np.max(np.abs(vals)))
    scale = 1.0
    offset = ""
    if peak > 0.0 and np.isfinite(peak):
        exp = int(np.floor(np.log10(peak)))
        if exp < -2 or exp > 3:
            scale = 10.0 ** exp
            offset = "×10" + str(exp).translate(_EXP_DIGITS)
    labels = []
    for value in vals:
        shown = float(value) / scale
        if abs(shown) < 1e-8:
            shown = 0.0
        labels.append(f"{shown:.4g}".replace("-", "−"))
    return labels, offset


def _minor_parts(step: float) -> int:
    if step <= 0.0 or not np.isfinite(step):
        return 5
    leading = step / 10.0 ** np.floor(np.log10(step))
    if abs(leading - 2.0) < 0.05 or abs(leading - 2.5) < 0.05:
        return 4
    return 5


def _minor_ticks(majors: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if majors.size < 2:
        return np.array([], dtype=float)
    out: list[float] = []

    def _fill(start: float, step: float, *, forward: bool) -> None:
        parts = _minor_parts(abs(step))
        delta = abs(step) / parts
        value = start
        for _ in range(32):
            value = value + delta if forward else value - delta
            if value <= lo or value >= hi:
                return
            out.append(value)

    for left, right in zip(majors[:-1], majors[1:]):
        step = float(right - left)
        parts = _minor_parts(step)
        for k in range(1, parts):
            value = float(left) + step * k / parts
            if lo < value < hi:
                out.append(value)
    _fill(float(majors[0]), float(majors[1] - majors[0]), forward=False)
    _fill(float(majors[-1]), float(majors[-1] - majors[-2]), forward=True)
    return np.asarray(out, dtype=float)


def color_axis_ticks(
    lo: float, hi: float, *, nbins: int = 6,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Major ticks, minor ticks, major labels, and the scientific offset.

    ``majors[0]`` and ``majors[-1]`` are always the exact ends of the bar.
    """
    lo_f = float(lo)
    hi_f = float(hi)
    empty = np.array([], dtype=float)
    if not np.isfinite(lo_f) or not np.isfinite(hi_f):
        return empty, empty, [], ""
    if hi_f < lo_f:
        lo_f, hi_f = hi_f, lo_f
    span = hi_f - lo_f
    if span <= 0.0:
        return np.array([lo_f]), empty, [], ""
    from matplotlib import ticker

    locator = ticker.MaxNLocator(
        nbins=max(2, int(nbins)),
        steps=[1, 2, 2.5, 5, 10],
        min_n_ticks=2,
    )
    nice = np.asarray(locator.tick_values(lo_f, hi_f), dtype=float)
    pad = span * 1e-6
    nice = nice[(nice >= lo_f - pad) & (nice <= hi_f + pad)]
    if lo_f < 0.0 < hi_f and not np.any(np.abs(nice) <= pad):
        nice = np.sort(np.append(nice, 0.0))
    minors = _minor_ticks(nice, lo_f, hi_f) if nice.size >= 2 else np.array([], dtype=float)
    inner = nice[(nice > lo_f + pad) & (nice < hi_f - pad)]
    majors = np.concatenate([[lo_f], inner, [hi_f]])
    labels, offset = _tick_labels(majors)
    return majors, minors, labels, offset


class _ColorAxis(QWidget):
    """Full-height color bar with major ticks, minor ticks, and a unit title."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._name = ""
        self._lo = float("nan")
        self._hi = float("nan")
        self._title = ""
        self._images: dict[tuple[str, int], QImage] = {}
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setFixedWidth(_AXIS_WIDTH)

    def sizeHint(self) -> QSize:
        return QSize(_AXIS_WIDTH, 480)

    def set_scale(self, name: str, lo: float, hi: float, title: str) -> None:
        lo_f = float(lo)
        hi_f = float(hi)
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f
        same = (
            name == self._name
            and title == self._title
            and np.isfinite(self._lo)
            and abs(lo_f - self._lo) <= 1e-9 * max(1.0, abs(lo_f))
            and abs(hi_f - self._hi) <= 1e-9 * max(1.0, abs(hi_f))
        )
        if same:
            return
        self._name = name
        self._lo = lo_f
        self._hi = hi_f
        self._title = title
        self.update()

    def _bar_image(self, height: int) -> QImage | None:
        if height < 2 or not self._name:
            return None
        key = (self._name, height)
        cached = self._images.get(key)
        if cached is not None:
            return cached
        rgb = np.clip(colormap_samples(self._name, height)[::-1], 0.0, 1.0)
        rgba = np.zeros((height, 1, 4), dtype=np.uint8)
        rgba[:, 0, :3] = (rgb * 255.0).astype(np.uint8)
        rgba[:, 0, 3] = 255
        rgba = np.ascontiguousarray(np.repeat(rgba, _AXIS_BAR_W, axis=1))
        image = QImage(
            rgba.data, _AXIS_BAR_W, height, int(rgba.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()
        self._images[key] = image
        if len(self._images) > 24:
            self._images.pop(next(iter(self._images)))
        return image

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        fm = painter.fontMetrics()
        lo, hi = self._lo, self._hi
        if not self._name or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            painter.fillRect(self.rect(), QColor(BG))
            painter.end()
            return
        # Painted as part of the view, so it shares the canvas zero color,
        # with a dark glass panel so the ticks read on any background.
        painter.fillRect(self.rect(), QColor.fromRgbF(*zero_rgb(self._name, lo, hi)))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(12, 14, 18, 170))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(2, 2, -2, -2), 6, 6)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        color = QColor.fromRgbF(*ink_rgb((0.0, 0.0, 0.0)))
        nbins = max(2, min(8, int(self.height() / 52)))
        majors, minors, labels, offset = color_axis_ticks(lo, hi, nbins=nbins)
        top = fm.height() // 2 + 4
        bottom = self.height() - (fm.height() // 2 + 4)
        if bottom - top < 8:
            painter.end()
            return
        bar = QRectF(_AXIS_PAD_X, top, _AXIS_BAR_W, bottom - top)
        image = self._bar_image(max(2, int(bar.height())))
        if image is not None:
            painter.drawImage(bar, image)
        pen = QPen(color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(bar)
        spine = bar.right()

        def y_of(value: float) -> float:
            return top + (hi - value) / (hi - lo) * (bottom - top)

        for value in minors:
            y = y_of(float(value))
            painter.drawLine(int(spine), int(round(y)), int(spine) + 4, int(round(y)))
        shown_y: list[float] = []
        # Ends first so the top and bottom labels always win a collision.
        order = [0, len(majors) - 1, *range(1, len(majors) - 1)]
        for idx in order:
            value = float(majors[idx])
            y = y_of(value)
            if any(abs(y - prev) < fm.height() for prev in shown_y):
                painter.drawLine(int(spine), int(round(y)), int(spine) + 5, int(round(y)))
                continue
            length = 9 if abs(value) <= (hi - lo) * 1e-6 else 7
            painter.drawLine(int(spine), int(round(y)), int(spine) + length, int(round(y)))
            painter.drawText(
                int(spine) + length + 4,
                int(round(y + fm.ascent() / 2 - 1)),
                labels[idx],
            )
            shown_y.append(y)
        title = f"{self._title}  {offset}".strip() if offset else self._title
        if title:
            title_x = self.width() - 4 - fm.height()
            painter.save()
            painter.translate(title_x + fm.height() / 2, (top + bottom) / 2)
            painter.rotate(-90)
            painter.drawText(
                QRectF(-bar.height() / 2, -fm.height() / 2, bar.height(), fm.height()),
                Qt.AlignmentFlag.AlignCenter,
                title,
            )
            painter.restore()
        painter.end()


class DoseVolumeWindow(VispyViewWindow):
    """3D dose volume with grain / XY / plan controls."""

    def __init__(
        self,
        session_ids: Sequence[str],
        base_dir: str,
        *,
        settings: ViewSettings | None = None,
        initial_preset: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title="Dose Volume (3D)", side_panel_default_width=420, parent=parent)
        self._updating = True
        # Before the canvas exists: moving a live GL widget can drop its context.
        self._mount_color_axis()
        gl = "gl+" if ensure_gl_plus() else None
        self._vispy_canvas = self.add_vispy_canvas(
            keys="interactive", size=(1200, 800), gl=gl,
        )
        self._scene = DoseScene(self._vispy_canvas)
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

        self._load_task = DebouncedBackgroundTask(debounce_ms=0, parent=self)
        self._load_task.finished.connect(self._on_load_finished)

        self._show_status("Loading dose data…")
        self._start_load()

    def _build_controls(self) -> QWidget:
        panel, layout = make_side_panel_column()
        layout.addWidget(
            make_presets_menu_button(
                [(p.id, p.label, True) for p in PRESETS],
                self._apply_preset,
            )
        )

        # Compare is what the picture answers, including the dose quantity.
        # Beam, phantom, and view follow.
        self._session_group = QGroupBox("Session")
        self._session_layout = QVBoxLayout(self._session_group)
        self._session_layout.setSpacing(2)
        self._session_buttons = QButtonGroup(self)
        self._session_buttons.idClicked.connect(self._on_session_picked)
        self._listed_ids: list[str] = []
        self._session_group.setVisible(False)
        layout.addWidget(self._session_group)

        compare_layout = self._add_group(layout, "Compare")
        self._show_combo = self._add_segment(
            compare_layout, "Show",
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
        layout.addWidget(self._gamma_group)

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
        self._smear_spin = self._add_spin(
            beam_layout, "Energy spread", 0.0, 10.0, 0.1, DEFAULT_ENERGY_SPREAD_PCT,
            decimals=2,
        )
        self._smear_spin.setSuffix(" %")
        self._smear_spin.setToolTip(
            "Beam energy spread σE, percent of energy. Range straggle is added on top."
        )
        self._plane_combo = self._add_segment(
            beam_layout, "Plane",
            ((SIGMA_PLANE_CHAMBER, "Chamber"), (SIGMA_PLANE_ISO, "Isocenter")),
            self._on_controls_changed,
        )
        self._plane_combo.set_button_tooltips({
            SIGMA_PLANE_CHAMBER: "Keep the measured spot σ at the chamber.",
            SIGMA_PLANE_ISO: "Project spot σ to isocenter by SAD / SDD.",
        })
        self._plane_combo.set_current(DEFAULT_SIGMA_PLANE)
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
            PLAN_SIGMA_MEASURED: "This session's σ, one value per energy layer.",
            PLAN_SIGMA_REFERENCE: "Per-layer σ from another loaded session.",
            PLAN_SIGMA_INTERLOCK: "The interlock σ.",
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
            view_layout, "Ray", _RAY_ITEMS, self._on_ray_changed,
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
        self._add_row(view_layout, "Sample", self._interp)
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
        self._add_row(view_layout, "Grid", self._grid_label)
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
        layout.addWidget(color_group)

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
        layout.addStretch(1)
        return panel

    def _mount_color_axis(self) -> None:
        """Glue the full-height scale to the right edge of the view, left of the splitter handle."""
        self._color_axis = _ColorAxis()
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        # The zero-colored background runs edge to edge, so no gutter around the canvas.
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._splitter.insertWidget(0, wrap)
        # Reparent the plot host straight into the row so Qt never drops it on a null parent.
        row.addWidget(self._plot_host, stretch=1)
        row.addWidget(self._color_axis)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

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

    def _density_unit(self) -> str:
        per_area = self._ray_combo.currentData() == RAY_INTEGRAL
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
            sigma_plane=self._choice(self._plane_combo),
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
            self._set_combo(self._plane_combo, config.sigma_plane)
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
            rate = gamma_pass_rate(self._scene.gamma_pass)
            return title if rate is None else f"{title} · {rate:.1f} % pass"
        density = self._density_unit()
        if self._drawn_difference():
            if percent:
                return "meas − plan (% of peak)"
            return f"meas − plan ({density})"
        return density

    def _update_legend(self) -> None:
        compare = self._drawn_difference()
        name = active_scale(compare, self._scale_combo.currentData() or DEFAULT_SCALE)
        if self._scene.gamma:
            name = GAMMA_CMAP
        percent = compare and self._error_combo.currentData() == ERROR_PERCENT
        lo, hi = self._legend_numbers()
        self._color_axis.set_scale(name, lo, hi, self._legend_title(percent=percent))

    def _on_controls_changed(self, *_args) -> None:
        if self._updating:
            return
        self._schedule_refresh()

    def _on_plan_sigma_changed(self, *_args) -> None:
        self._ref_row.setVisible(self._choice(self._plan_sigma_combo) == PLAN_SIGMA_REFERENCE)
        self._on_controls_changed()

    def _show_gamma_verdict(self) -> None:
        if not self._scene.gamma:
            self._gamma_label.setText(self._no_plan_reason() if self._gamma_mode() else "—")
            self._gamma_label.setStyleSheet("")
            return
        tally = self._scene.gamma_pass
        rate = gamma_pass_rate(tally)
        verdict, color = gamma_verdict(rate)
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
        self._load_task.schedule(loader)

    @Slot(int, object)
    def _on_load_finished(self, gen: int, result: object) -> None:
        if gen != self._load_task.generation:
            return
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

    def _start_refresh(self) -> None:
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
        )
        self._show_field_extent()
        self._phantom_note.setText(self._scene.phantom_note or "—")
        self._grid_label.setText(self._scene.volume_note or "—")
        self._maybe_seed_absolute_window()
        self._update_legend()
        self._show_gamma_verdict()
        n_meas = 0 if measured_batch is None else int(measured_batch.center.shape[0])
        spots = f"{n_meas:,}"
        if residual:
            spots += f" · plan {int(plan_batch.center.shape[0]):,}"
        if config.overlay_plan and not residual:
            spots += " · no plan"
        if measured_batch is not None and not has_dose:
            spots += " · relative" if config.grain == GRAIN_TIMESLICE else " · plan MU"
        self._spots_label.setText(spots)

    def _show_field_extent(self) -> None:
        ext = getattr(self._scene, "field_extent", None)
        self._field_size.setText("—" if ext is None else f"{ext[0]:.0f} × {ext[1]:.0f} × {ext[2]:.0f} mm")

    def _no_plan_reason(self) -> str:
        if self._xy_combo.currentData() == XY_PLAN:
            return "XY is Plan, so there is no separate plan to compare"
        return "No plan to compare against"


def run_dose_volume_window(
    session_ids: Sequence[str],
    base_dir: str = "test_data",
    *,
    settings: ViewSettings | None = None,
    initial_preset: str | None = None,
) -> None:
    if not session_ids:
        return
    run_view_window(
        lambda: DoseVolumeWindow(
            session_ids,
            base_dir,
            settings=settings,
            initial_preset=initial_preset,
        ),
        maximize=True,
    )
