"""Qt shell for the Distribution Explorer viewer."""

from __future__ import annotations

from typing import Any, Sequence

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..common.segmented_control import SegmentedControl
from ..common.settings import ViewSettings
from .async_refresh import DebouncedBackgroundTask
from .distribution_catalog import (
    MODE_BY_ID,
    PRESETS,
    PRESET_BY_ID,
    VIEW_OPTIONS,
    DistributionConfig,
    metric_source_for_mode,
    resolve_mode_id,
)
from .distribution_data import (
    clear_load_cache,
    default_mode,
    load_sessions_for_mode,
    probe_mode_availability,
)
from .distribution_ui import render_distribution
from .plot_view_shell import (
    PlotViewWindow,
    make_presets_menu_button,
    make_side_panel_column,
    run_view_window,
)
from .unified_catalog import DATA_SOURCE_SPOT
from .unified_view_controls import DataSourceOptionPanel


class DistributionExplorerWindow(PlotViewWindow):
    """Configurable density-distribution and fit-quality plots."""

    def __init__(
        self,
        session_ids: Sequence[str],
        base_dir: str,
        *,
        settings: ViewSettings | None = None,
        initial_preset: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            title="Distribution Explorer",
            figsize=(16, 9),
            side_panel_min_width=240,
            side_panel_default_width=300,
            parent=parent,
        )
        self._session_ids = list(session_ids)
        self._base_dir = base_dir
        self._settings = settings or ViewSettings()
        self._cache: dict[str, dict[str, Any]] = {}
        self._mode_available = probe_mode_availability(self._session_ids, self._base_dir)
        self._option_panel: DataSourceOptionPanel | None = None
        self._bg_segmented: SegmentedControl | None = None
        self._plot_style_segmented: SegmentedControl | None = None
        self._plot_style: str = "contour"
        self._refresh_generation = 0

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(60)
        self._refresh_timer.timeout.connect(self._start_refresh)

        self._load_task = DebouncedBackgroundTask(debounce_ms=0, parent=self)
        self._load_task.finished.connect(self._on_load_finished)

        self.set_side_panel(self._build_controls())
        if initial_preset and initial_preset in PRESET_BY_ID:
            self._apply_preset(initial_preset)
        else:
            mode = default_mode(
                self._session_ids,
                self._base_dir,
                availability=self._mode_available,
            )
            if self._option_panel is not None:
                mapping = metric_source_for_mode(mode)
                if mapping is not None:
                    metric_id, source = mapping
                    self._option_panel.select_id(metric_id, source=source)
            self._schedule_refresh()

    def _build_controls(self) -> QWidget:
        panel, layout = make_side_panel_column()

        layout.addWidget(
            make_presets_menu_button(
                [
                    (preset.id, preset.label, self._mode_available.get(preset.mode, False))
                    for preset in PRESETS
                ],
                self._apply_preset,
            )
        )

        self._option_panel = DataSourceOptionPanel(
            on_selection_changed=self._schedule_refresh,
        )
        self._option_panel.configure(
            VIEW_OPTIONS,
            self._mode_available,
            group_title="Distribution",
            preferred_source=DATA_SOURCE_SPOT,
        )
        layout.addWidget(self._option_panel)

        opts = QGroupBox("Options")
        opt_layout = QVBoxLayout(opts)
        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("BG Subtract"))
        self._bg_segmented = SegmentedControl([("off", "Off"), ("on", "On")])
        self._bg_segmented.selectionChanged.connect(self._on_bg_segment_changed)
        bg_row.addWidget(self._bg_segmented)
        bg_row.addStretch(1)
        opt_layout.addLayout(bg_row)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Plot Style"))
        self._plot_style_segmented = SegmentedControl(
            [("contour", "Contour"), ("scatter", "Scatter")],
        )
        self._plot_style_segmented.selectionChanged.connect(
            self._on_plot_style_segment_changed,
        )
        style_row.addWidget(self._plot_style_segmented)
        style_row.addStretch(1)
        opt_layout.addLayout(style_row)

        layout.addWidget(opts)

        layout.addStretch(1)
        return panel

    def _current_mode(self) -> str | None:
        if self._option_panel is None:
            return None
        metric_id = self._option_panel.selected_id()
        if metric_id is None:
            return None
        return resolve_mode_id(metric_id, self._option_panel.selected_source())

    def _sync_bg_subtract_controls(self) -> None:
        if self._bg_segmented is None:
            return
        mode_id = self._current_mode()
        mode_def = MODE_BY_ID.get(mode_id) if mode_id else None
        enabled = mode_def is None or mode_def.uses_bg_subtract
        self._bg_segmented.setEnabled(enabled)
        self._bg_segmented.set_current("on" if self._settings.bg_subtract else "off")

    def _sync_plot_style_controls(self) -> None:
        if self._plot_style_segmented is None:
            return
        mode_id = self._current_mode()
        mode_def = MODE_BY_ID.get(mode_id) if mode_id else None
        enabled = mode_def is None or mode_def.supports_plot_style
        self._plot_style_segmented.setEnabled(enabled)
        self._plot_style_segmented.set_current(self._plot_style)

    def _apply_preset(self, preset_id: str) -> None:
        preset = PRESET_BY_ID[preset_id]
        mapping = metric_source_for_mode(preset.mode)
        if self._option_panel is not None and mapping is not None:
            metric_id, source = mapping
            self._option_panel.select_id(metric_id, source=source)
        self._schedule_refresh()

    def _on_plot_style_segment_changed(self, key: str) -> None:
        if key == self._plot_style:
            return
        self._plot_style = key
        self._schedule_refresh()

    def _on_bg_segment_changed(self, key: str) -> None:
        checked = key == "on"
        if checked == self._settings.bg_subtract:
            return
        self._settings = ViewSettings(
            bg_subtract=checked,
            calibration_mode=self._settings.calibration_mode,
            cal_factors=self._settings.cal_factors,
        )
        self._cache.clear()
        clear_load_cache()
        self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        self._refresh_generation += 1
        self._refresh_timer.start()

    def _start_refresh(self) -> None:
        gen = self._refresh_generation
        mode = self._current_mode()
        if mode is None:
            return
        self._sync_bg_subtract_controls()
        self._sync_plot_style_controls()
        config = DistributionConfig(mode=mode, plot_style=self._plot_style)  # type: ignore[arg-type]
        self.setWindowTitle(config.title)

        if mode in self._cache:
            self._draw_plot(config, self._cache[mode])
            return

        session_ids = list(self._session_ids)
        base_dir = self._base_dir
        settings = self._settings

        def loader() -> tuple[int, str, dict[str, Any]]:
            data = load_sessions_for_mode(
                mode,
                session_ids,
                base_dir,
                settings=settings,
            )
            return gen, mode, data

        self._load_task.schedule(loader)

    @Slot(int, object)
    def _on_load_finished(self, _task_generation: int, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 3:
            return
        gen, mode, session_data = result
        if gen != self._refresh_generation:
            return
        if self._current_mode() != mode:
            return
        self._cache[mode] = session_data
        self._draw_plot(
            DistributionConfig(mode=mode, plot_style=self._plot_style),  # type: ignore[arg-type]
            session_data,
        )

    def _draw_plot(
        self,
        config: DistributionConfig,
        session_data: dict[str, Any],
    ) -> None:
        render_distribution(
            self.figure,
            config,
            session_data,
            self._base_dir,
        )
        self.draw_idle()


def run_distribution_explorer_window(
    session_ids: Sequence[str],
    base_dir: str = "test_data",
    *,
    settings: ViewSettings | None = None,
    initial_preset: str | None = None,
) -> None:
    if not session_ids:
        print("No sessions selected")
        return

    run_view_window(
        lambda: DistributionExplorerWindow(
            session_ids,
            base_dir,
            settings=settings,
            initial_preset=initial_preset,
        ),
    )
