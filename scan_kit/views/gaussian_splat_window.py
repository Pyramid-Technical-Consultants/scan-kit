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
    DEFAULT_AGREEMENT,
    DEFAULT_GAIN,
    DEFAULT_GANTRY_DEG,
    DEFAULT_SMEAR_MEV,
    DEFAULT_SPLAT_CAP,
    GRAIN_SPOT,
    GRAIN_TIMESLICE,
    MEDIUM_AIR,
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
    (MEDIUM_AIR, "Air"),
)
_AGREE_ITEMS = (
    (AGREE_TRANSPARENT, "Transparent"),
    (AGREE_WHITE, "White"),
)


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

    def set_energy_range(self, vmin: float | None, vmax: float | None) -> None:
        self._bar.setPixmap(self._energy_pm)
        self._mid.setText("Energy (MeV)\nyellow = high")
        if vmin is None or vmax is None:
            self._hi.setText("—")
            self._lo.setText("—")
            return
        self._hi.setText(f"{vmax:.1f}  high")
        self._lo.setText(f"{vmin:.1f}  low")

    def set_residual(self, zero: str = AGREE_TRANSPARENT) -> None:
        key = AGREE_WHITE if zero == AGREE_WHITE else AGREE_TRANSPARENT
        self._bar.setPixmap(self._resid_pm[key])
        mid = "agree = white" if key == AGREE_WHITE else "agree = transparent"
        self._mid.setText(f"meas − plan\n{mid}")
        self._hi.setText("hot  measured > plan")
        self._lo.setText("cold  plan > measured")


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

        self._legend_group = QGroupBox("Sessions")
        self._legend_layout = QVBoxLayout(self._legend_group)
        layout.addWidget(self._legend_group)

        source_group = QGroupBox("Source")
        source_layout = QVBoxLayout(source_group)
        source_layout.addWidget(QLabel("Grain"))
        self._grain_combo = QComboBox()
        for value, label in _GRAIN_ITEMS:
            self._grain_combo.addItem(label, value)
        self._grain_combo.currentIndexChanged.connect(self._on_grain_changed)
        source_layout.addWidget(self._grain_combo)
        source_layout.addWidget(QLabel("XY"))
        self._xy_combo = QComboBox()
        for value, label in _XY_ITEMS:
            self._xy_combo.addItem(label, value)
        self._xy_combo.currentIndexChanged.connect(self._on_controls_changed)
        source_layout.addWidget(self._xy_combo)
        self._overlay_plan = QCheckBox("Difference vs plan (hot / cold)")
        self._overlay_plan.toggled.connect(self._on_overlay_toggled)
        source_layout.addWidget(self._overlay_plan)
        source_layout.addWidget(QLabel("Agreement"))
        self._agree_combo = QComboBox()
        for value, label in _AGREE_ITEMS:
            self._agree_combo.addItem(label, value)
        self._set_combo(self._agree_combo, DEFAULT_AGREEMENT)
        self._agree_combo.currentIndexChanged.connect(self._on_agreement_changed)
        self._agree_combo.setEnabled(False)
        source_layout.addWidget(self._agree_combo)
        source_layout.addWidget(QLabel("Depth medium"))
        self._medium_combo = QComboBox()
        for value, label in _MEDIUM_ITEMS:
            self._medium_combo.addItem(label, value)
        self._medium_combo.currentIndexChanged.connect(self._on_controls_changed)
        source_layout.addWidget(self._medium_combo)
        layout.addWidget(source_group)

        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        self._gantry_spin = self._add_spin(
            display_layout, "Gantry (deg)", 0.0, 360.0, 5.0, DEFAULT_GANTRY_DEG,
            decimals=1,
        )
        self._gantry_spin.setWrapping(True)
        self._gantry_spin.setSuffix(" °")
        self._gain_spin = self._add_spin(
            display_layout, "Gain", 0.01, 100.0, 0.1, DEFAULT_GAIN, decimals=2,
        )
        self._smear_spin = self._add_spin(
            display_layout, "σz smear (MeV)", 0.01, 50.0, 0.1, DEFAULT_SMEAR_MEV,
            decimals=2,
        )
        display_layout.addWidget(QLabel("Splat cap"))
        self._cap_spin = QSpinBox()
        self._cap_spin.setRange(1_000, 5_000_000)
        self._cap_spin.setSingleStep(50_000)
        self._cap_spin.setValue(DEFAULT_SPLAT_CAP)
        self._cap_spin.valueChanged.connect(self._on_controls_changed)
        display_layout.addWidget(self._cap_spin)
        display_layout.addWidget(QLabel("Color"))
        self._energy_legend = _ColorLegend()
        display_layout.addWidget(self._energy_legend)
        layout.addWidget(display_group)

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

    def _read_config(self) -> SplatConfig:
        return SplatConfig(
            grain=self._grain_combo.currentData(),
            xy_mode=self._xy_combo.currentData(),
            overlay_plan=self._overlay_plan.isChecked(),
            agreement=self._agree_combo.currentData(),
            gain=self._gain_spin.value(),
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
            self._agree_combo.setEnabled(config.overlay_plan)
            self._set_combo(self._medium_combo, config.medium)
            self._gain_spin.setValue(config.gain)
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
        self._agree_combo.setEnabled(self._overlay_plan.isChecked())
        self._schedule_refresh()

    def _on_agreement_changed(self, *_args) -> None:
        if self._updating:
            return
        mode = self._agree_combo.currentData()
        self._scene.set_agreement(mode)
        if self._residual_active:
            self._energy_legend.set_residual(mode)

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
            self._energy_legend.set_residual(config.agreement)
        elif measured_batch is not None and measured_batch.energy_mev.size:
            e = measured_batch.energy_mev
            finite = e[np.isfinite(e)]
            if finite.size:
                self._energy_legend.set_energy_range(
                    float(np.min(finite)), float(np.max(finite)),
                )
            else:
                self._energy_legend.set_energy_range(None, None)
        else:
            self._energy_legend.set_energy_range(None, None)
        if gen != self._refresh_generation:
            return
        self._scene.render(
            measured_batch, plan_batch, axis,
            gain=config.gain, gantry_deg=config.gantry_deg,
            residual=residual, agreement=config.agreement,
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
