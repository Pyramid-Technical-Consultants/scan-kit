"""Qt shell for the Distribution Explorer viewer."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..common.ic_xy_distribution import normalize_contour_cutoff_percentile
from ..common.data_filter import FILTER_ALL, FILTER_BEAM_BOTH, FILTER_BEAM_ON
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
    new_headless_figure,
    run_view_window,
)
from .unified_catalog import (
    DATA_SOURCE_SPOT,
    DATA_SOURCE_TIMESLICE,
    DISTRIBUTION_PLOT_STYLES,
    PLOT_STYLE_CONTOUR,
)
from .unified_view_controls import (
    DataFilterPanel,
    DataSourceOptionPanel,
    PlotStylePanel,
    sync_data_filter_panel,
)


_CONTOUR_CUTOFF_OPTION = "contour_cutoff"
_log = logging.getLogger(__name__)


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
        self._plot_style_panel: PlotStylePanel | None = None
        self._filter_panel: DataFilterPanel | None = None
        self._refresh_generation = 0

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(80)
        self._refresh_timer.timeout.connect(self._start_refresh)

        self._load_task = DebouncedBackgroundTask(debounce_ms=0, parent=self)
        self._load_task.finished.connect(self._on_load_finished)

        self._render_task = DebouncedBackgroundTask(debounce_ms=50, parent=self)
        self._render_task.finished.connect(self._on_render_finished)

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
            self._sync_data_filter_for_mode()
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
            on_selection_changed=self._on_metric_changed,
        )
        self._option_panel.configure(
            VIEW_OPTIONS,
            self._mode_available,
            group_title="Distribution",
            preferred_source=DATA_SOURCE_SPOT,
        )
        layout.addWidget(self._option_panel)

        self._plot_style_panel = PlotStylePanel(
            DISTRIBUTION_PLOT_STYLES,
            on_selection_changed=self._schedule_refresh,
            current=PLOT_STYLE_CONTOUR,
        )
        self._plot_style_panel.add_percent_spinbox(
            _CONTOUR_CUTOFF_OPTION,
            "Contour Cutoff",
            value=int(round(self._settings.contour_cutoff_percentile)),
        )
        layout.addWidget(self._plot_style_panel)

        self._filter_panel = DataFilterPanel(
            on_selection_changed=self._schedule_refresh,
            domain_current=FILTER_ALL,
            beam_current=FILTER_BEAM_ON,
        )
        layout.addWidget(self._filter_panel)

        layout.addStretch(1)
        return panel

    def _current_mode(self) -> str | None:
        if self._option_panel is None:
            return None
        metric_id = self._option_panel.selected_id()
        if metric_id is None:
            return None
        return resolve_mode_id(metric_id, self._option_panel.selected_source())

    def _read_config(self) -> DistributionConfig:
        mode = self._current_mode() or PRESETS[0].mode
        plot_style = (
            self._plot_style_panel.selected_key()
            if self._plot_style_panel is not None
            else PLOT_STYLE_CONTOUR
        )
        cutoff = self._settings.contour_cutoff_percentile
        if self._plot_style_panel is not None:
            spin_val = self._plot_style_panel.spin_value(_CONTOUR_CUTOFF_OPTION)
            if spin_val is not None:
                cutoff = float(spin_val)
        cutoff = normalize_contour_cutoff_percentile(cutoff)
        return DistributionConfig(
            mode=mode,
            plot_style=plot_style or PLOT_STYLE_CONTOUR,  # type: ignore[arg-type]
            contour_cutoff_percentile=cutoff,
            domain_filter=(
                self._filter_panel.selected_domain()
                if self._filter_panel is not None
                else FILTER_ALL
            ) or FILTER_ALL,
            beam_state_filter=(
                self._filter_panel.selected_beam_state()
                if self._filter_panel is not None
                else FILTER_BEAM_BOTH
            ) or FILTER_BEAM_BOTH,
        )

    def _mode_filter_state(self) -> tuple[bool, bool]:
        mode_id = self._current_mode()
        mode = MODE_BY_ID.get(mode_id) if mode_id else None
        if mode is None:
            return True, False
        supports_filter = mode.supports_data_filter
        has_beam_state = supports_filter and mode.source == DATA_SOURCE_TIMESLICE
        return supports_filter, has_beam_state

    def _sync_data_filter_for_mode(self) -> None:
        supports_filter, has_beam_state = self._mode_filter_state()
        sync_data_filter_panel(
            self._filter_panel,
            supports_filter=supports_filter,
            has_beam_state=has_beam_state,
        )

    def _on_metric_changed(self) -> None:
        self._sync_data_filter_for_mode()
        self._schedule_refresh()

    def _sync_display_controls(self) -> None:
        mode_id = self._current_mode()
        mode_def = MODE_BY_ID.get(mode_id) if mode_id else None
        supports_style = mode_def is None or mode_def.supports_plot_style
        if self._plot_style_panel is not None:
            self._plot_style_panel.set_enabled(supports_style)
        panel_supports_filter, has_beam_state = self._mode_filter_state()
        sync_data_filter_panel(
            self._filter_panel,
            supports_filter=panel_supports_filter if mode_def is not None else True,
            has_beam_state=has_beam_state,
        )
        if self._plot_style_panel is not None:
            contour_mode = (
                supports_style
                and self._plot_style_panel.selected_key() == PLOT_STYLE_CONTOUR
            )
            self._plot_style_panel.set_option_visible(
                _CONTOUR_CUTOFF_OPTION, contour_mode,
            )

    def _schedule_render(self, config: DistributionConfig, session_data: dict) -> None:
        gen = self._refresh_generation
        figsize = tuple(float(v) for v in self.figure.get_size_inches())
        base_dir = self._base_dir

        def render_fn() -> tuple[int, Figure | None]:
            fig = new_headless_figure(figsize)
            try:
                render_distribution(fig, config, session_data, base_dir)
            except Exception:
                _log.exception("Distribution Explorer render failed")
                return gen, None
            return gen, fig

        self._render_task.schedule(render_fn)

    def _apply_preset(self, preset_id: str) -> None:
        preset = PRESET_BY_ID[preset_id]
        mapping = metric_source_for_mode(preset.mode)
        if self._option_panel is not None and mapping is not None:
            metric_id, source = mapping
            self._option_panel.select_id(metric_id, source=source)
        self._schedule_refresh()

    def _persist_contour_cutoff(self, cutoff: float) -> None:
        cutoff = normalize_contour_cutoff_percentile(cutoff)
        if cutoff == self._settings.contour_cutoff_percentile:
            return
        self._settings = ViewSettings(
            bg_subtract=self._settings.bg_subtract,
            calibration_mode=self._settings.calibration_mode,
            cal_factors=self._settings.cal_factors,
            selected_sessions=list(self._settings.selected_sessions),
            contour_cutoff_percentile=cutoff,
        )
        try:
            self._settings.save(self._base_dir)
        except OSError:
            pass

    def _schedule_refresh(self) -> None:
        self._refresh_generation += 1
        self._refresh_timer.start()

    def _start_refresh(self) -> None:
        gen = self._refresh_generation
        config = self._read_config()
        mode = config.mode
        self._sync_display_controls()
        self._persist_contour_cutoff(config.contour_cutoff_percentile)
        self.setWindowTitle(config.title)

        if mode in self._cache:
            self._schedule_render(config, self._cache[mode])
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
        self._schedule_render(self._read_config(), session_data)

    @Slot(int, object)
    def _on_render_finished(self, _task_generation: int, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 2:
            return
        gen, fig = result
        if gen != self._refresh_generation:
            if isinstance(fig, Figure):
                import matplotlib.pyplot as plt
                plt.close(fig)
            return
        if not isinstance(fig, Figure):
            return
        previous = self.figure
        self.figure = fig
        self.canvas.figure = fig
        self.draw_idle()
        if previous is not fig:
            import matplotlib.pyplot as plt

            plt.close(previous)


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
