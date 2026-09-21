"""Qt shell for the 3D Gaussian splat viewer (visPy)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..common import ViewSettings
from ..common.plotting import format_session_legend_label
from ..common.session_notes import load_notes
from .async_refresh import DebouncedBackgroundTask
from .gaussian_splat_catalog import (
    AGREE_TRANSPARENT,
    AGREE_WHITE,
    COLOR_ENERGY,
    COLOR_MU,
    COLOR_PROTONS,
    DEFAULT_AGREEMENT,
    DEFAULT_COLOR,
    DEFAULT_ERROR_MODE,
    DEFAULT_ERROR_MU,
    DEFAULT_ERROR_PCT,
    DEFAULT_ERROR_SCALE,
    DEFAULT_GAIN,
    DEFAULT_GANTRY_DEG,
    DEFAULT_IC_GAP_MM,
    DEFAULT_SMEAR_MEV,
    DEFAULT_SPLAT_CAP,
    ERROR_ABSOLUTE,
    ERROR_PERCENT,
    GRAIN_SPOT,
    GRAIN_TIMESLICE,
    MEDIUM_COPPER,
    MEDIUM_WATER,
    PLAN_RGB,
    PRESET_BY_ID,
    PRESETS,
    XY_IC1,
    XY_IC2,
    XY_ISO_RAY,
    XY_PLAN,
    SplatConfig,
)
from .gaussian_splat_data import (
    build_view_batches,
    color_legend_spec,
    load_splat_sessions,
    range_axis_for_medium,
)
from .gaussian_splat_visual import residual_agreement_rgba
from .gaussian_splat_vispy import SplatScene, default_session_colors
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
_AGREE_ITEMS = (
    (AGREE_TRANSPARENT, "Transparent"),
    (AGREE_WHITE, "White"),
)
_ERROR_ITEMS = (
    (ERROR_PERCENT, "Percent of plan"),
    (ERROR_ABSOLUTE, "Absolute MU"),
)
_COLOR_ITEMS = (
    (COLOR_ENERGY, "Energy"),
    (COLOR_MU, "Dose (MU)"),
    (COLOR_PROTONS, "Protons"),
)
_GAIN_SLIDER_MAX = 100


def _viridis_bar_pixmap(width: int = 16, height: int = 120) -> QPixmap:
    from matplotlib import cm

    t = np.linspace(1.0, 0.0, height, dtype=np.float32)
    rgba = np.ascontiguousarray((cm.viridis(t) * 255.0).astype(np.uint8))
    img = np.repeat(rgba[:, None, :], width, axis=1)
    qimg = QImage(
        img.data, width, height, int(img.strides[0]), QImage.Format.Format_RGBA8888,
    )
    return QPixmap.fromImage(qimg.copy())


def _residual_bar_pixmap(zero: str, width: int = 16, height: int = 120) -> QPixmap:
    t = np.linspace(1.0, 0.0, height, dtype=np.float64)
    rgba = residual_agreement_rgba(t, 1.0 - t, zero=zero)
    img = np.ascontiguousarray(
        np.repeat((np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)[:, None, :], width, axis=1),
    )
    qimg = QImage(
        img.data, width, height, int(img.strides[0]), QImage.Format.Format_RGBA8888,
    )
    return QPixmap.fromImage(qimg.copy())


class _ColorLegend(QWidget):
    """Viridis energy scale, or red/blue residual (hot / cold)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._energy_pm = _viridis_bar_pixmap()
        self._resid_pm = {
            AGREE_TRANSPARENT: _residual_bar_pixmap(AGREE_TRANSPARENT),
            AGREE_WHITE: _residual_bar_pixmap(AGREE_WHITE),
        }
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._bar = QLabel()
        self._bar.setPixmap(self._energy_pm)
        self._bar.setFixedSize(16, 120)
        self._bar.setScaledContents(True)
        labels = QVBoxLayout()
        labels.setContentsMargins(0, 0, 0, 0)
        self._hi = QLabel("—")
        self._mid = QLabel("Energy (MeV)\nyellow = high")
        self._mid.setWordWrap(True)
        self._lo = QLabel("—")
        labels.addWidget(self._hi)
        labels.addStretch(1)
        labels.addWidget(self._mid)
        labels.addStretch(1)
        labels.addWidget(self._lo)
        layout.addWidget(self._bar)
        layout.addLayout(labels, stretch=1)

    def set_scalar_range(
        self,
        vmin: float | None,
        vmax: float | None,
        *,
        label: str = "Energy (MeV)\nyellow = high",
        fmt: str = ".1f",
    ) -> None:
        self._bar.setPixmap(self._energy_pm)
        self._mid.setText(label)
        if vmin is None or vmax is None:
            self._hi.setText("—")
            self._lo.setText("—")
            return
        self._hi.setText(format(vmax, fmt))
        self._lo.setText(format(vmin, fmt))

    def set_energy_range(self, vmin: float | None, vmax: float | None) -> None:
        self.set_scalar_range(vmin, vmax)

    def set_residual(
        self,
        zero: str = AGREE_TRANSPARENT,
        mode: str = DEFAULT_ERROR_MODE,
        scale: float = DEFAULT_ERROR_SCALE,
    ) -> None:
        key = AGREE_WHITE if zero == AGREE_WHITE else AGREE_TRANSPARENT
        self._bar.setPixmap(self._resid_pm[key])
        agree = "agree = white" if key == AGREE_WHITE else "agree = transparent"
        if mode == ERROR_PERCENT:
            pct = max(float(scale), 0.0) * 100.0
            self._hi.setText(f"+{pct:.0f}%")
            self._lo.setText(f"−{pct:.0f}%")
            self._mid.setText(f"meas − plan\n% of plan\n{agree}")
        else:
            mu = max(float(scale), 0.0)
            self._hi.setText(f"+{mu:.2f} MU")
            self._lo.setText(f"−{mu:.2f} MU")
            self._mid.setText(f"meas − plan\nabsolute MU\n{agree}")


class GaussianSplatWindow(VispyViewWindow):
    """3D Gaussian splat viewer with grain / XY / plan controls."""

    def __init__(
        self,
        session_ids: Sequence[str],
        base_dir: str,
        *,
        settings: ViewSettings | None = None,
        initial_preset: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title="Gaussian Splats (3D)", parent=parent)
        self._updating = True
        gl = "gl+" if ensure_gl_plus() else None
        self._vispy_canvas = self.add_vispy_canvas(
            keys="interactive", size=(1200, 800), gl=gl,
        )
        self._scene = SplatScene(self._vispy_canvas)
        self.set_side_panel(self._build_controls())

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

        self._show_status("Loading Gaussian data…")
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

        compare_group = QGroupBox("vs Plan")
        compare_layout = QVBoxLayout(compare_group)
        self._overlay_plan = QCheckBox("Show difference (hot / cold)")
        self._overlay_plan.toggled.connect(self._on_overlay_toggled)
        compare_layout.addWidget(self._overlay_plan)
        self._agree_combo = self._add_combo(
            compare_layout, "Where they agree", _AGREE_ITEMS, self._on_agreement_changed,
        )
        self._set_combo(self._agree_combo, DEFAULT_AGREEMENT)
        self._error_combo = self._add_combo(
            compare_layout, "Error", _ERROR_ITEMS, self._on_error_mode_changed,
        )
        self._set_combo(self._error_combo, DEFAULT_ERROR_MODE)
        compare_layout.addWidget(QLabel("Color range"))
        self._error_scale_spin = QDoubleSpinBox()
        self._error_scale_spin.valueChanged.connect(self._on_error_scale_changed)
        self._configure_error_scale_spin(DEFAULT_ERROR_MODE, DEFAULT_ERROR_SCALE)
        compare_layout.addWidget(self._error_scale_spin)
        self._set_compare_enabled(False)
        layout.addWidget(compare_group)

        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        self._color_combo = self._add_combo(
            display_layout, "Color", _COLOR_ITEMS, self._on_color_mode_changed,
        )
        self._set_combo(self._color_combo, DEFAULT_COLOR)
        self._gap_spin = self._add_spin(
            display_layout, "IC gap", 0.1, 100.0, 0.5, DEFAULT_IC_GAP_MM,
            decimals=1,
        )
        self._gap_spin.setSuffix(" mm")
        self._gap_spin.setEnabled(False)
        self._energy_legend = _ColorLegend()
        display_layout.addWidget(self._energy_legend)
        gain_row = QHBoxLayout()
        gain_row.setContentsMargins(0, 0, 0, 0)
        gain_row.addWidget(QLabel("Gain"))
        self._gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._gain_slider.setRange(0, _GAIN_SLIDER_MAX)
        self._gain_slider.setValue(int(round(DEFAULT_GAIN * _GAIN_SLIDER_MAX)))
        self._gain_slider.setToolTip("Display brightness")
        self._gain_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._gain_label = QLabel(f"{DEFAULT_GAIN:.2f}")
        self._gain_label.setMinimumWidth(32)
        self._gain_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        self._gain_slider.valueChanged.connect(self._on_gain_changed)
        gain_row.addWidget(self._gain_slider, stretch=1)
        gain_row.addWidget(self._gain_label)
        display_layout.addLayout(gain_row)
        self._smear_spin = self._add_spin(
            display_layout, "σz smear", 0.01, 50.0, 0.1, DEFAULT_SMEAR_MEV,
            decimals=2,
        )
        display_layout.addWidget(QLabel("Splat cap"))
        self._cap_spin = QSpinBox()
        self._cap_spin.setRange(1_000, 5_000_000)
        self._cap_spin.setSingleStep(50_000)
        self._cap_spin.setValue(DEFAULT_SPLAT_CAP)
        self._cap_spin.valueChanged.connect(self._on_controls_changed)
        display_layout.addWidget(self._cap_spin)
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

    def _add_combo(
        self,
        layout: QVBoxLayout,
        label: str,
        items: tuple[tuple[str, str], ...],
        handler,
    ) -> QComboBox:
        layout.addWidget(QLabel(label))
        combo = QComboBox()
        for value, text in items:
            combo.addItem(text, value)
        combo.currentIndexChanged.connect(handler)
        layout.addWidget(combo)
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
        layout.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.valueChanged.connect(self._on_controls_changed)
        layout.addWidget(spin)
        return spin

    def _gain_value(self) -> float:
        return self._gain_slider.value() / float(_GAIN_SLIDER_MAX)

    def _error_scale_value(self) -> float:
        raw = float(self._error_scale_spin.value())
        if self._error_combo.currentData() == ERROR_PERCENT:
            return raw / 100.0
        return raw

    def _configure_error_scale_spin(self, mode: str, scale: float) -> None:
        if mode == ERROR_PERCENT:
            self._error_scale_spin.setDecimals(0)
            self._error_scale_spin.setRange(1.0, 100.0)
            self._error_scale_spin.setSingleStep(1.0)
            self._error_scale_spin.setSuffix(" %")
            self._error_scale_spin.setValue(max(1.0, min(100.0, float(scale) * 100.0)))
        else:
            self._error_scale_spin.setDecimals(2)
            self._error_scale_spin.setRange(0.01, 10.0)
            self._error_scale_spin.setSingleStep(0.05)
            self._error_scale_spin.setSuffix(" MU")
            self._error_scale_spin.setValue(max(0.01, min(10.0, float(scale))))

    def _set_compare_enabled(self, on: bool) -> None:
        self._agree_combo.setEnabled(on)
        self._error_combo.setEnabled(on)
        self._error_scale_spin.setEnabled(on)

    def _read_config(self) -> SplatConfig:
        return SplatConfig(
            grain=self._grain_combo.currentData(),
            xy_mode=self._xy_combo.currentData(),
            overlay_plan=self._overlay_plan.isChecked(),
            agreement=self._agree_combo.currentData(),
            error_mode=self._error_combo.currentData(),
            error_scale=self._error_scale_value(),
            color_mode=self._color_combo.currentData(),
            ic_gap_mm=self._gap_spin.value(),
            gain=self._gain_value(),
            smear_axis_units=self._smear_spin.value(),
            splat_cap=self._cap_spin.value(),
            gantry_deg=self._gantry_spin.value(),
            medium=self._medium_combo.currentData(),
        )

    def _set_combo(self, combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _set_config(self, config: SplatConfig) -> None:
        self._updating = True
        try:
            self._set_combo(self._grain_combo, config.grain)
            self._set_combo(self._xy_combo, config.xy_mode)
            self._overlay_plan.setChecked(config.overlay_plan)
            self._set_combo(self._agree_combo, config.agreement)
            self._set_combo(self._error_combo, config.error_mode)
            self._configure_error_scale_spin(config.error_mode, config.error_scale)
            self._set_compare_enabled(config.overlay_plan)
            self._set_combo(self._medium_combo, config.medium)
            self._set_combo(self._color_combo, config.color_mode)
            self._gap_spin.setValue(config.ic_gap_mm)
            self._gap_spin.setEnabled(config.color_mode == COLOR_PROTONS)
            self._gain_slider.setValue(
                int(round(max(0.0, min(1.0, config.gain)) * _GAIN_SLIDER_MAX)),
            )
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

    def _on_overlay_toggled(self, *_args) -> None:
        if self._updating:
            return
        self._set_compare_enabled(self._overlay_plan.isChecked())
        self._schedule_refresh()

    def _on_agreement_changed(self, *_args) -> None:
        if self._updating:
            return
        self._scene.set_agreement(self._agree_combo.currentData())
        self._update_residual_legend()

    def _on_error_mode_changed(self, *_args) -> None:
        if self._updating:
            return
        mode = self._error_combo.currentData()
        default = DEFAULT_ERROR_PCT / 100.0 if mode == ERROR_PERCENT else DEFAULT_ERROR_MU
        self._updating = True
        try:
            self._configure_error_scale_spin(mode, default)
        finally:
            self._updating = False
        self._apply_error_metric()

    def _on_error_scale_changed(self, *_args) -> None:
        if self._updating:
            return
        self._apply_error_metric()

    def _apply_error_metric(self) -> None:
        mode = self._error_combo.currentData()
        scale = self._error_scale_value()
        self._scene.set_error_metric(mode, scale)
        self._update_residual_legend()

    def _update_residual_legend(self) -> None:
        if not self._residual_active:
            return
        self._energy_legend.set_residual(
            self._agree_combo.currentData(),
            self._error_combo.currentData(),
            self._error_scale_value(),
        )

    def _on_color_mode_changed(self, *_args) -> None:
        if self._updating:
            return
        self._gap_spin.setEnabled(self._color_combo.currentData() == COLOR_PROTONS)
        self._schedule_refresh()

    def _on_gain_changed(self, *_args) -> None:
        gain = self._gain_value()
        self._gain_label.setText(f"{gain:.2f}")
        if self._updating:
            return
        self._scene.set_gain(gain)

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

        self._show_status("Loading Gaussian data…")
        self._load_task.schedule(loader)

    @Slot(int, object)
    def _on_load_finished(self, gen: int, result: object) -> None:
        if gen != self._load_task.generation:
            return
        if not isinstance(result, dict):
            self._show_status("Failed to load Gaussian data")
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
            plan_rgb=PLAN_RGB,
        )

        self._smear_spin.setSuffix(f" {axis.smear_label}")
        self._info_label.setText(
            f"{n_used:,} splats drawn\n"
            f"{n_raw:,} raw samples\n"
            f"{axis.axis_label}"
        )
        residual = bool(config.overlay_plan and plan_batch is not None and plan_batch.center.size)
        self._residual_active = residual
        if residual:
            self._energy_legend.set_residual(
                config.agreement, config.error_mode, config.error_scale,
            )
        elif measured_batch is not None:
            label, fmt = color_legend_spec(config.color_mode)
            self._energy_legend.set_scalar_range(
                measured_batch.color_lo, measured_batch.color_hi, label=label, fmt=fmt,
            )
        else:
            self._energy_legend.set_scalar_range(None, None)
        if gen != self._refresh_generation:
            return
        self._scene.render(
            measured_batch, plan_batch, axis,
            gain=config.gain, gantry_deg=config.gantry_deg,
            residual=residual, agreement=config.agreement,
            error_mode=config.error_mode, error_scale=config.error_scale,
        )


def run_gaussian_splat_window(
    session_ids: Sequence[str],
    base_dir: str = "test_data",
    *,
    settings: ViewSettings | None = None,
    initial_preset: str | None = None,
) -> None:
    if not session_ids:
        return
    run_view_window(
        lambda: GaussianSplatWindow(
            session_ids,
            base_dir,
            settings=settings,
            initial_preset=initial_preset,
        ),
        maximize=True,
    )
