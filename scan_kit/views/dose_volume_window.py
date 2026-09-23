"""Qt shell for the 3D dose-volume viewer (visPy)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..common import ViewSettings
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
    DEFAULT_IC_GAP_MM,
    DEFAULT_SMEAR_MEV,
    DEFAULT_SPOT_CAP,
    DEFAULT_WEIGHT,
    ERROR_ABSOLUTE,
    ERROR_PERCENT,
    GRAIN_SPOT,
    GRAIN_TIMESLICE,
    MEDIUM_COPPER,
    MEDIUM_WATER,
    PRESET_BY_ID,
    PRESETS,
    RAY_INTEGRAL,
    RAY_MAXIMUM,
    RAY_TRANSPARENT,
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
from .dose_volume_fill import colormap_samples
from .dose_volume_raycast import manual_color_limits, suggest_abs_window
from .dose_volume_vispy import DoseScene, default_session_colors
from .plot_view_shell import (
    VispyViewWindow,
    make_presets_menu_button,
    make_side_panel_column,
    run_view_window,
)
from .vispy_plot import ensure_gl_plus

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
    (MEDIUM_COPPER, "Copper"),
)
_SHOW_ITEMS = (
    ("dose", "Dose"),
    ("difference", "Difference"),
)
_ERROR_ITEMS = (
    (ERROR_PERCENT, "Percent of peak"),
    (ERROR_ABSOLUTE, "Absolute"),
)
_WEIGHT_ITEMS = (
    (WEIGHT_MU, "Dose (MU)"),
    (WEIGHT_PROTONS, "Protons"),
)
_RAY_ITEMS = (
    (RAY_INTEGRAL, "Integrate"),
    (RAY_MAXIMUM, "Maximum"),
    (RAY_TRANSPARENT, "Transparent"),
)
_GAIN_SLIDER_MAX = 100
_SCALE_SLIDER_MAX = 1000
_LABEL_WIDTH = 100


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
_AXIS_BAR_W = 18
# Fixed so tick text and the ×10 offset never shove the controls sideways.
_AXIS_WIDTH = 176
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
    """Major ticks, minor ticks, major labels, and the scientific offset."""
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
    majors = np.asarray(locator.tick_values(lo_f, hi_f), dtype=float)
    pad = span * 1e-6
    majors = majors[(majors >= lo_f - pad) & (majors <= hi_f + pad)]
    if lo_f < 0.0 < hi_f and not np.any(np.abs(majors) <= pad):
        majors = np.sort(np.append(majors, 0.0))
    if majors.size == 0:
        majors = np.array([lo_f, hi_f])
    labels, offset = _tick_labels(majors)
    return majors, _minor_ticks(majors, lo_f, hi_f), labels, offset


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
        painter.fillRect(self.rect(), self.palette().color(QPalette.ColorRole.Window))
        color = self.palette().color(QPalette.ColorRole.WindowText)
        fm = painter.fontMetrics()
        lo, hi = self._lo, self._hi
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            painter.end()
            return
        nbins = max(2, min(8, int(self.height() / 52)))
        majors, minors, labels, offset = color_axis_ticks(lo, hi, nbins=nbins)
        top = fm.height() + 2 if offset else fm.height() // 2 + 4
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
        for value, label in zip(majors, labels):
            y = y_of(float(value))
            if any(abs(y - prev) < fm.height() for prev in shown_y):
                painter.drawLine(int(spine), int(round(y)), int(spine) + 5, int(round(y)))
                continue
            length = 9 if abs(float(value)) <= (hi - lo) * 1e-6 else 7
            painter.drawLine(int(spine), int(round(y)), int(spine) + length, int(round(y)))
            painter.drawText(
                int(spine) + length + 4,
                int(round(y + fm.ascent() / 2 - 1)),
                label,
            )
            shown_y.append(y)
        if offset:
            painter.drawText(int(spine) + 11, fm.ascent() + 1, offset)
        if self._title:
            title_x = self.width() - 6 - fm.height()
            painter.save()
            painter.translate(title_x + fm.height() / 2, (top + bottom) / 2)
            painter.rotate(-90)
            painter.drawText(
                QRectF(-bar.height() / 2, -fm.height() / 2, bar.height(), fm.height()),
                Qt.AlignmentFlag.AlignCenter,
                self._title,
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
        super().__init__(title="Dose Volume (3D)", parent=parent)
        self._updating = True
        gl = "gl+" if ensure_gl_plus() else None
        self._vispy_canvas = self.add_vispy_canvas(
            keys="interactive", size=(1200, 800), gl=gl,
        )
        self._scene = DoseScene(self._vispy_canvas)
        self.set_side_panel(self._build_controls())
        self._mount_color_axis()
        self._update_legend()
        self._scene.range_listener = self._on_gpu_range

        self._session_ids = list(session_ids)
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

        beam_group = QGroupBox("Beam")
        beam_layout = QVBoxLayout(beam_group)
        beam_layout.setSpacing(4)
        self._grain_combo = self._add_combo(
            beam_layout, "Grain", _GRAIN_ITEMS, self._on_grain_changed,
        )
        self._xy_combo = self._add_combo(
            beam_layout, "XY", _XY_ITEMS, self._on_controls_changed,
        )
        self._medium_combo = self._add_combo(
            beam_layout, "Depth medium", _MEDIUM_ITEMS, self._on_controls_changed,
        )
        self._gantry_spin = self._add_spin(
            beam_layout, "Gantry", 0.0, 360.0, 5.0, DEFAULT_GANTRY_DEG,
            decimals=1,
        )
        self._gantry_spin.setWrapping(True)
        self._gantry_spin.setSuffix(" °")
        layout.addWidget(beam_group)

        self._seq_scale = DEFAULT_SCALE
        self._div_scale = DEFAULT_DIVERGENT_SCALE
        color_group = QGroupBox("Color")
        color_layout = QVBoxLayout(color_group)
        color_layout.setSpacing(4)
        self._show_combo = self._add_combo(
            color_layout, "Show", _SHOW_ITEMS, self._on_show_changed,
        )
        self._show_combo.setToolTip(
            "Dose paints the measured field. Difference paints measured minus plan."
        )
        self._ray_combo = self._add_combo(
            color_layout, "Ray", _RAY_ITEMS, self._on_ray_changed,
        )
        self._ray_combo.setToolTip(
            "How each viewing ray combines the field. Integrate sums the whole "
            "ray. Maximum keeps the hottest sample. Transparent fades like fog."
        )
        self._set_combo(self._ray_combo, DEFAULT_RAY)
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
        self._auto_check.setToolTip(
            "Fit the color scale to the rays on screen. Dose uses the brightest "
            "ray. Difference uses the darkest and the brightest."
        )
        gain_row.addWidget(self._gain_slider, stretch=1)
        gain_row.addWidget(self._error_scale_spin)
        gain_row.addWidget(self._gain_label)
        gain_row.addWidget(self._auto_check)
        color_layout.addWidget(self._gain_box)
        self._auto_check.toggled.connect(self._on_auto_changed)
        self._auto_check.setChecked(True)
        layout.addWidget(color_group)

        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(4)
        self._weight_combo = self._add_combo(
            display_layout, "Deposit", _WEIGHT_ITEMS, self._on_weight_mode_changed,
        )
        self._set_combo(self._weight_combo, DEFAULT_WEIGHT)
        self._gap_spin = self._add_spin(
            display_layout, "IC gap", 0.1, 100.0, 0.5, DEFAULT_IC_GAP_MM,
            decimals=1,
        )
        self._gap_spin.setSuffix(" mm")
        self._gap_spin.setEnabled(False)
        self._smear_spin = self._add_spin(
            display_layout, "σz smear", 0.01, 50.0, 0.1, DEFAULT_SMEAR_MEV,
            decimals=2,
        )
        self._cap_spin = QSpinBox()
        self._cap_spin.setRange(1_000, 5_000_000)
        self._cap_spin.setSingleStep(50_000)
        self._cap_spin.setValue(DEFAULT_SPOT_CAP)
        self._cap_spin.valueChanged.connect(self._on_controls_changed)
        self._add_row(display_layout, "Spot cap", self._cap_spin)
        layout.addWidget(display_group)

        self._legend_group = QGroupBox("Sessions")
        self._legend_layout = QVBoxLayout(self._legend_group)
        layout.addWidget(self._legend_group)

        info_group = QGroupBox("Loaded")
        info_layout = QVBoxLayout(info_group)
        self._info_label = QLabel("—")
        self._info_label.setWordWrap(True)
        self._info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse,
        )
        info_layout.addWidget(self._info_label)
        layout.addWidget(info_group)
        layout.addStretch(1)
        return panel

    def _mount_color_axis(self) -> None:
        """Park the scale beside the controls so it can use the full height."""
        self._color_axis = _ColorAxis()
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        row.addWidget(self._color_axis)
        # Reparent the controls directly so Qt never drops them on a null parent.
        row.addWidget(self._side_scroll, stretch=1)
        self._splitter.addWidget(wrap)
        bar = self._color_axis.sizeHint().width()
        side = max(self._side_default_width, self._side_min_width)
        total = max(self.width(), side + bar + 640)
        self._splitter.setSizes([max(total - side - bar, 400), side + bar])

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
            return WEIGHT_MU
        return combo.currentData()

    def _abs_suffix(self) -> str:
        protons = self._deposit_weight() == WEIGHT_PROTONS
        per_area = self._ray_combo.currentData() == RAY_INTEGRAL
        unit = "" if protons else " MU"
        return f"{unit}/mm²" if per_area else f"{unit}/mm³"

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
        return self._show_combo.currentData() == "difference"

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
        return DoseVolumeConfig(
            grain=self._grain_combo.currentData(),
            xy_mode=self._xy_combo.currentData(),
            overlay_plan=compare,
            error_mode=self._error_combo.currentData(),
            error_scale=self._error_scale_value(),
            weight_mode=self._weight_combo.currentData(),
            ic_gap_mm=self._gap_spin.value(),
            gain=self._gain_value(),
            smear_axis_units=self._smear_spin.value(),
            splat_cap=self._cap_spin.value(),
            gantry_deg=self._gantry_spin.value(),
            medium=self._medium_combo.currentData(),
            ray_mode=self._ray_combo.currentData(),
            scale=active_scale(compare, self._scale_combo.currentData() or DEFAULT_SCALE),
            auto_scale=self._auto_check.isChecked(),
        )

    def _set_combo(self, combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _set_config(self, config: DoseVolumeConfig) -> None:
        self._updating = True
        try:
            self._set_combo(self._grain_combo, config.grain)
            self._set_combo(self._xy_combo, config.xy_mode)
            self._set_combo(self._show_combo, "difference" if config.overlay_plan else "dose")
            allowed = {name for name, _label in scales_for(config.overlay_plan)}
            if config.scale in allowed:
                if config.overlay_plan:
                    self._div_scale = config.scale
                else:
                    self._seq_scale = config.scale
            self._fill_scale_combo(config.overlay_plan)
            self._set_combo(self._error_combo, config.error_mode)
            self._configure_error_scale_spin(config.error_mode, config.error_scale)
            self._set_combo(self._medium_combo, config.medium)
            self._set_combo(self._weight_combo, config.weight_mode)
            self._set_combo(self._ray_combo, config.ray_mode)
            self._auto_check.setChecked(config.auto_scale)
            self._gain = max(0.0, min(1.0, float(config.gain)))
            self._sync_color_controls()
            self._gap_spin.setValue(config.ic_gap_mm)
            self._gap_spin.setEnabled(config.weight_mode == WEIGHT_PROTONS)
            self._smear_spin.setValue(config.smear_axis_units)
            self._cap_spin.setValue(config.splat_cap)
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
        self._gap_spin.setEnabled(self._weight_combo.currentData() == WEIGHT_PROTONS)
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

    def _legend_numbers(self) -> tuple[float, float]:
        auto = self._auto_check.isChecked()
        percent = (
            self._comparing()
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
            difference=self._comparing(),
            transparent=mode == RAY_TRANSPARENT,
            integral=integral,
            gain=self._gain_value(),
            typical=float(self._scene.dose_peak),
            ray_scale=float(self._scene.ray_peak) if integral else float(self._scene.dose_peak),
            error_scale=self._error_scale_value(),
            absolute=self._error_combo.currentData() == ERROR_ABSOLUTE,
        )

    def _legend_title(self, *, percent: bool) -> str:
        if self._comparing():
            if percent:
                return "meas − plan (%)"
            if self._weight_combo.currentData() == WEIGHT_PROTONS:
                return "meas − plan"
            return "meas − plan (MU)"
        protons = self._weight_combo.currentData() == WEIGHT_PROTONS
        unit = "protons" if protons else "MU"
        per_area = self._ray_combo.currentData() == RAY_INTEGRAL
        return f"{unit} / mm²" if per_area else f"{unit} / mm³"

    def _update_legend(self) -> None:
        compare = self._comparing()
        name = active_scale(compare, self._scale_combo.currentData() or DEFAULT_SCALE)
        percent = compare and self._error_combo.currentData() == ERROR_PERCENT
        lo, hi = self._legend_numbers()
        self._color_axis.set_scale(name, lo, hi, self._legend_title(percent=percent))

    def _on_controls_changed(self, *_args) -> None:
        if self._updating:
            return
        self._schedule_refresh()

    def _update_session_legend(self, loaded_ids: list[str], colors: list[str]) -> None:
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not loaded_ids:
            self._legend_group.setVisible(False)
            return
        self._legend_group.setVisible(len(loaded_ids) > 1)
        for sid, color in zip(loaded_ids, colors):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(f"background-color: {color}; border: 1px solid #666;")
            label = QLabel(format_session_legend_label(sid, self._notes))
            row_layout.addWidget(swatch)
            row_layout.addWidget(label, stretch=1)
            self._legend_layout.addWidget(row)

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
        config = self._read_config()
        if gen != self._refresh_generation:
            return
        self.setWindowTitle(config.title)
        loaded_ids = [sid for sid in self._session_ids if sid in self._sources]
        colors = default_session_colors(len(loaded_ids))
        self._update_session_legend(loaded_ids, colors)
        measured_batch, plan_batch, axis, n_raw, n_used = build_view_batches(
            self._sources,
            loaded_ids,
            config,
            self._base_dir,
            plan_rgb=(1.0, 1.0, 1.0),
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
            difference=residual, agreement=config.agreement,
            error_mode=config.error_mode, error_scale=config.error_scale,
            weight_mode=config.weight_mode, ic_gap_mm=config.ic_gap_mm,
            smear=config.smear_axis_units, ray_mode=config.ray_mode,
            scale=config.scale, auto_scale=config.auto_scale,
        )
        self._maybe_seed_absolute_window()
        self._update_legend()
        info = (
            f"{n_used:,} spots\n"
            f"{n_raw:,} raw samples\n"
            f"{axis.axis_label}"
        )
        if self._scene.volume_note:
            info += f"\n{self._scene.volume_note}"
        if config.overlay_plan and plan_batch is not None and not has_dose:
            info += "\nMeasured volume uses plan MU (no spot dose)"
        self._info_label.setText(info)


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
