"""Qt shell for the universal binned summary viewer."""

from __future__ import annotations

from typing import Sequence

from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..common import ViewSettings
from ..common.ic_xy_distribution import normalize_contour_cutoff_percentile
from ..common.data_filter import FILTER_ALL, FILTER_BEAM_BOTH, FILTER_BEAM_ON
from .binned_summary_catalog import (
    DATA_SOURCE_SPOT_ISO,
    DATA_SOURCE_TIMESLICE_ISO,
    GLYPH_BOX,
    GLYPH_CONTOUR,
    GLYPH_MEAN,
    GLYPH_SCATTER,
    GLYPH_VIOLIN,
    PRESETS,
    PRESET_BY_ID,
    VIEW_OPTIONS,
    X_ENERGY,
    X_PARAM_BY_ID,
    X_PARAMS,
    Y_CURRENT_RATIO,
    Y_DOSE_RATE,
    Y_IC_CURRENT,
    Y_IC12_POS_DIFF,
    Y_SIGMA,
    Y_SIGMA_ERROR,
    Y_GROUP_BY_ID,
    BinnedSummaryConfig,
)
from ..data.types import data_source_is_timeslice
from .binned_summary_data import (
    BINNED_REGISTRY_SOURCE_IDS,
    available_x_params_for_source,
    default_config,
    load_sessions_current_ratios,
    load_sessions_dose_rate,
    load_sessions_ic_current,
    load_sessions_for_source,
    load_sessions_sigma_error,
    probe_view_option_availability,
)
from ..data.availability import probe_sessions
from .async_refresh import DebouncedBackgroundTask
from .binned_summary_ui import render_binned_summary
from .plot_view_shell import (
    PlotViewWindow,
    make_presets_menu_button,
    make_side_panel_column,
    new_headless_figure,
    run_view_window,
)
from .unified_catalog import BINNED_PLOT_STYLES, option_key
from .unified_view_controls import (
    DataFilterPanel,
    DataSourceOptionPanel,
    HistogramPanel,
    PlotStylePanel,
    sync_data_filter_panel,
)

_OPT_TREND = "trend"
_OPT_CORR = "corr"
_OPT_FLIERS = "fliers"
_OPT_CONTOUR_CUTOFF = "contour_cutoff"


class BinnedSummaryWindow(PlotViewWindow):
    """Configurable binned summary: pick Y metric group, X binning, and glyph."""

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
            title="Binned Summary",
            figsize=(16, 9),
            side_panel_min_width=240,
            side_panel_default_width=300,
            parent=parent,
        )
        self._session_ids = list(session_ids)
        self._base_dir = base_dir
        self._settings = settings
        self._session_data_cache: dict[tuple, dict[str, dict]] = {}
        self._registry_y_cache: dict[str, dict[str, dict]] = {}
        self._spot_data: dict[str, dict] = {}
        self._registry_availability: dict[str, bool] = {}
        self._option_availability: dict[str, bool] = {}
        self._initial_load_done = False
        self._x_avail: set[str] = set()
        self._updating = False
        self._metric_panel: DataSourceOptionPanel | None = None
        self._plot_style_panel: PlotStylePanel | None = None
        self._histogram_panel: HistogramPanel | None = None
        self._filter_panel: DataFilterPanel | None = None
        self._pending_preset = initial_preset
        self._refresh_generation = 0
        self._presets_button: QToolButton | None = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(60)
        self._refresh_timer.timeout.connect(self._start_refresh)

        self._load_task = DebouncedBackgroundTask(debounce_ms=0, parent=self)
        self._load_task.finished.connect(self._on_initial_load_finished)

        self._render_task = DebouncedBackgroundTask(debounce_ms=50, parent=self)
        self._render_task.finished.connect(self._on_render_finished)

        self.set_side_panel(self._build_controls())
        self._show_status_message("Loading summary data…")
        self._start_initial_load()

    def _show_status_message(self, message: str) -> None:
        self.figure.clear()
        self.figure.text(0.5, 0.5, message, ha="center", va="center")
        self.draw_idle()

    def _start_initial_load(self) -> None:
        session_ids = list(self._session_ids)
        base_dir = self._base_dir
        settings = self._settings

        def loader() -> tuple[
            dict[str, dict],
            dict[str, bool],
            dict[str, bool],
        ]:
            spot_data = load_sessions_for_source(
                session_ids,
                base_dir,
                DATA_SOURCE_SPOT_ISO,
                settings=settings,
            )
            registry_availability = probe_sessions(
                session_ids,
                base_dir,
                source_ids=BINNED_REGISTRY_SOURCE_IDS,
            )
            option_availability = probe_view_option_availability(
                session_ids,
                base_dir,
                spot_data=spot_data,
                registry_availability=registry_availability,
                settings=settings,
            )
            return spot_data, registry_availability, option_availability

        self._load_task.schedule(loader)

    @Slot(int, object)
    def _on_initial_load_finished(self, _task_generation: int, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 3:
            self._show_status_message("Failed to load summary data")
            return
        spot_data, registry_availability, option_availability = result
        self._spot_data = spot_data
        bg = self._settings.bg_subtract if self._settings else False
        self._session_data_cache[(DATA_SOURCE_SPOT_ISO, bg)] = spot_data
        self._registry_availability = registry_availability
        self._option_availability = option_availability
        self._initial_load_done = True

        self._refresh_presets_menu()

        if self._metric_panel is not None:
            self._metric_panel.configure(
                VIEW_OPTIONS,
                self._option_availability,
                group_title="Y Metric",
                preferred_source=DATA_SOURCE_SPOT_ISO,
            )

        if not self._spot_data and not self._option_availability:
            self._show_status_message("No summary data found for the selected sessions.")
            return

        preset = self._pending_preset
        self._pending_preset = None
        if preset and preset in PRESET_BY_ID:
            self._apply_preset(preset)
        else:
            cfg = default_config(
                self._spot_data,
                source=DATA_SOURCE_SPOT_ISO,
                option_availability=self._option_availability,
            )
            self._set_controls_from_config(cfg)
            self._sync_data_filter_for_metric()
            self._schedule_refresh()

    def _build_controls(self) -> QWidget:
        panel, layout = make_side_panel_column()

        self._presets_button = make_presets_menu_button(
            [
                (
                    preset.id,
                    preset.label,
                    self._preset_is_available(preset),
                )
                for preset in PRESETS
            ],
            self._apply_preset,
        )
        layout.addWidget(self._presets_button)

        self._metric_panel = DataSourceOptionPanel(
            on_selection_changed=self._on_metric_selection_changed,
        )
        self._metric_panel.configure(
            VIEW_OPTIONS,
            self._option_availability,
            group_title="Y Metric",
            preferred_source=DATA_SOURCE_SPOT_ISO,
        )
        layout.addWidget(self._metric_panel)

        x_group = QGroupBox("X Parameter")
        x_layout = QVBoxLayout(x_group)
        self._x_combo = QComboBox()
        self._x_model = QStandardItemModel(self._x_combo)
        self._x_combo.setModel(self._x_model)
        self._refresh_x_combo()
        self._x_combo.currentIndexChanged.connect(self._on_x_param_changed)
        x_layout.addWidget(self._x_combo)

        self._x_bins_row = QWidget()
        bins_layout = QHBoxLayout(self._x_bins_row)
        bins_layout.setContentsMargins(0, 0, 0, 0)
        bins_layout.addWidget(QLabel("Bins"))
        self._x_bins_spin = QSpinBox()
        self._x_bins_spin.setRange(2, 50)
        self._x_bins_spin.setValue(8)
        self._x_bins_spin.valueChanged.connect(self._on_controls_changed)
        bins_layout.addWidget(self._x_bins_spin, stretch=1)
        x_layout.addWidget(self._x_bins_row)
        self._sync_x_bins_control()
        layout.addWidget(x_group)

        cutoff_default = 5
        if self._settings is not None:
            cutoff_default = int(round(self._settings.contour_cutoff_percentile))
        self._plot_style_panel = PlotStylePanel(
            BINNED_PLOT_STYLES,
            on_selection_changed=self._on_controls_changed,
            current=GLYPH_VIOLIN,
        )
        self._plot_style_panel.add_checkbox(_OPT_TREND, "Trend Line", checked=True)
        self._plot_style_panel.add_percent_spinbox(
            _OPT_CONTOUR_CUTOFF,
            "Contour Cutoff",
            value=cutoff_default,
        )
        self._plot_style_panel.add_checkbox(_OPT_CORR, "Correlation Panel")
        self._plot_style_panel.add_checkbox(_OPT_FLIERS, "Show Box Outliers")
        layout.addWidget(self._plot_style_panel)
        self._sync_plot_style_controls()

        self._histogram_panel = HistogramPanel(
            on_selection_changed=self._on_controls_changed,
        )
        layout.addWidget(self._histogram_panel)

        self._filter_panel = DataFilterPanel(
            on_selection_changed=self._on_controls_changed,
            domain_current=FILTER_ALL,
            beam_current=FILTER_BEAM_ON,
        )
        layout.addWidget(self._filter_panel)

        if self._initial_load_done and not self._spot_data and not self._option_availability:
            note = QLabel("No summary data found for the selected sessions.")
            note.setWordWrap(True)
            layout.addWidget(note)

        layout.addStretch(1)
        return panel

    def _session_cache_key(self, source: str) -> tuple:
        bg = self._settings.bg_subtract if self._settings else False
        return (source, bg)

    def _load_registry_y_group(self, y_group: str) -> dict[str, dict]:
        cached = self._registry_y_cache.get(y_group)
        if cached is not None:
            return cached
        if y_group == Y_DOSE_RATE:
            loaded = load_sessions_dose_rate(self._session_ids, self._base_dir)
        elif y_group == Y_CURRENT_RATIO:
            loaded = load_sessions_current_ratios(
                self._session_ids, self._base_dir, settings=self._settings,
            )
        elif y_group == Y_IC_CURRENT:
            loaded = load_sessions_ic_current(
                self._session_ids, self._base_dir, settings=self._settings,
            )
        else:
            loaded = {}
        self._registry_y_cache[y_group] = loaded
        return loaded

    def _session_data_for_y_group(self, y_group: str) -> dict[str, dict]:
        if y_group in {Y_DOSE_RATE, Y_CURRENT_RATIO, Y_IC_CURRENT}:
            return self._load_registry_y_group(y_group)
        if y_group == Y_SIGMA_ERROR:
            source = self._current_source()
            cache_key = (*self._session_cache_key(source), Y_SIGMA_ERROR)
            cached = self._session_data_cache.get(cache_key)
            if cached is not None:
                return cached
            loaded = load_sessions_sigma_error(
                self._session_ids,
                self._base_dir,
                source,  # type: ignore[arg-type]
                settings=self._settings,
            )
            self._session_data_cache[cache_key] = loaded
            return loaded
        source = self._current_source()
        cache_key = self._session_cache_key(source)
        cached = self._session_data_cache.get(cache_key)
        if cached is not None:
            return cached
        loaded = load_sessions_for_source(
            self._session_ids,
            self._base_dir,
            source,  # type: ignore[arg-type]
            settings=self._settings,
        )
        self._session_data_cache[cache_key] = loaded
        if not data_source_is_timeslice(source):  # type: ignore[arg-type]
            self._spot_data = loaded
        return loaded

    def _spot_session_data(self) -> dict[str, dict]:
        cache_key = self._session_cache_key(DATA_SOURCE_SPOT_ISO)
        cached = self._session_data_cache.get(cache_key)
        if cached is not None:
            return cached
        loaded = load_sessions_for_source(
            self._session_ids,
            self._base_dir,
            DATA_SOURCE_SPOT_ISO,
            settings=self._settings,
        )
        self._session_data_cache[cache_key] = loaded
        self._spot_data = loaded
        return loaded

    def _x_avail_for_y_group(self, y_group: str) -> set[str]:
        group = Y_GROUP_BY_ID.get(y_group)
        if group is None:
            return set()
        source = group.sources[0]
        session_data = self._session_data_for_y_group(y_group)
        return available_x_params_for_source(session_data, source)  # type: ignore[arg-type]

    def _preset_is_available(self, preset) -> bool:
        group = Y_GROUP_BY_ID[preset.y_group]
        key = option_key(group.sources[0], preset.y_group)
        if not bool(self._option_availability.get(key, False)):
            return False
        if not self._initial_load_done:
            return False
        return preset.x_param in self._x_avail_for_y_group(preset.y_group)

    def _refresh_presets_menu(self) -> None:
        button = self._presets_button
        if button is None:
            return
        menu = button.menu()
        if menu is None:
            return
        actions = menu.actions()
        for action, preset in zip(actions, PRESETS, strict=False):
            action.setEnabled(bool(self._preset_is_available(preset)))
        button.setEnabled(any(bool(self._preset_is_available(p)) for p in PRESETS))

    def _current_source(self) -> str:
        if self._metric_panel is None:
            return DATA_SOURCE_SPOT_ISO
        return self._metric_panel.selected_source()

    def _session_data(self) -> dict[str, dict]:
        y_group = self._metric_panel.selected_id() if self._metric_panel is not None else None
        if y_group is None:
            return self._spot_session_data()
        return self._session_data_for_y_group(y_group)

    def _refresh_x_combo(self) -> None:
        source = self._current_source()
        session_data = self._session_data()
        self._x_avail = available_x_params_for_source(session_data, source)  # type: ignore[arg-type]
        current = self._x_combo.currentData()
        self._updating = True
        try:
            self._x_model.clear()
            for param in X_PARAMS:
                item = QStandardItem(param.label)
                item.setData(param.id, Qt.ItemDataRole.UserRole)
                item.setEnabled(param.id in self._x_avail)
                self._x_model.appendRow(item)
            if current in self._x_avail:
                idx = self._x_combo.findData(current)
            elif X_ENERGY in self._x_avail:
                idx = self._x_combo.findData(X_ENERGY)
            else:
                idx = 0 if self._x_model.rowCount() else -1
            if idx >= 0:
                self._x_combo.setCurrentIndex(idx)
        finally:
            self._updating = False

    def _current_x_param(self):
        x_id = self._x_combo.currentData()
        if x_id is None:
            return None
        return X_PARAM_BY_ID.get(str(x_id))

    def _sync_x_bins_control(self) -> None:
        x_param = self._current_x_param()
        quantile = x_param is not None and x_param.bin_mode == "quantile"
        linear_glyph = self._selected_glyph() in (GLYPH_SCATTER, GLYPH_CONTOUR)
        show_bins = quantile and not linear_glyph
        self._x_bins_row.setVisible(show_bins)
        self._x_bins_spin.setEnabled(show_bins)

    def _sync_plot_style_controls(self) -> None:
        panel = self._plot_style_panel
        if panel is None:
            return
        glyph = self._selected_glyph()
        panel.set_option_visible(_OPT_CONTOUR_CUTOFF, glyph == GLYPH_CONTOUR)
        panel.set_option_visible(
            _OPT_FLIERS,
            glyph in (GLYPH_BOX, GLYPH_VIOLIN, GLYPH_MEAN),
        )

    def _on_x_param_changed(self, *_args) -> None:
        if self._updating:
            return
        x_param = self._current_x_param()
        if x_param is not None and x_param.bin_mode == "quantile":
            self._x_bins_spin.setValue(x_param.n_bins)
        self._sync_x_bins_control()
        self._on_controls_changed()

    def _selected_glyph(self) -> str:
        if self._plot_style_panel is None:
            return GLYPH_VIOLIN
        return self._plot_style_panel.selected_key() or GLYPH_VIOLIN

    def _read_config(self) -> BinnedSummaryConfig:
        y_group = (
            self._metric_panel.selected_id()
            if self._metric_panel is not None
            else None
        )
        panel = self._plot_style_panel
        hist_panel = self._histogram_panel
        x_param = self._current_x_param()
        n_bins = None
        if x_param is not None and x_param.bin_mode == "quantile":
            n_bins = self._x_bins_spin.value()
        cutoff = 5.0
        if panel is not None:
            spin_val = panel.spin_value(_OPT_CONTOUR_CUTOFF)
            if spin_val is not None:
                cutoff = float(spin_val)
        cutoff = normalize_contour_cutoff_percentile(cutoff)
        x_param_obj = self._current_x_param()
        x_param_id = (
            x_param_obj.id
            if x_param_obj is not None
            else self._x_combo.currentData() or X_ENERGY
        )
        return BinnedSummaryConfig(
            y_group=y_group or PRESETS[0].y_group,
            source=self._current_source(),
            x_param=str(x_param_id),
            glyph=self._selected_glyph(),  # type: ignore[arg-type]
            show_trend=panel.is_checked(_OPT_TREND) if panel else True,
            show_hist=hist_panel.is_enabled() if hist_panel else False,
            hist_bin_count=hist_panel.bin_count() if hist_panel else 30,
            hist_shared_bins=hist_panel.shared_bins() if hist_panel else False,
            show_corr=panel.is_checked(_OPT_CORR) if panel else False,
            show_fliers=panel.is_checked(_OPT_FLIERS) if panel else False,
            contour_cutoff_percentile=cutoff,
            n_bins=n_bins,
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

    def _set_controls_from_config(self, config: BinnedSummaryConfig) -> None:
        self._updating = True
        try:
            if self._metric_panel is not None:
                self._metric_panel.select_id(
                    config.y_group,
                    source=config.source,  # type: ignore[arg-type]
                )
            self._refresh_x_combo()
            x_idx = self._x_combo.findData(config.x_param)
            if x_idx >= 0:
                self._x_combo.setCurrentIndex(x_idx)
            if config.n_bins is not None:
                self._x_bins_spin.setValue(config.n_bins)
            else:
                x_param = X_PARAM_BY_ID.get(config.x_param)
                if x_param is not None and x_param.bin_mode == "quantile":
                    self._x_bins_spin.setValue(x_param.n_bins)
            self._sync_x_bins_control()
            if self._plot_style_panel is not None:
                self._plot_style_panel.set_current(config.glyph)
                self._plot_style_panel.set_checked(_OPT_TREND, config.show_trend)
                self._plot_style_panel.set_checked(_OPT_CORR, config.show_corr)
                self._plot_style_panel.set_checked(_OPT_FLIERS, config.show_fliers)
                self._plot_style_panel.set_spin_value(
                    _OPT_CONTOUR_CUTOFF,
                    int(round(config.contour_cutoff_percentile)),
                )
            self._sync_plot_style_controls()
            if self._histogram_panel is not None:
                self._histogram_panel.set_from_config(
                    show_hist=config.show_hist,
                    hist_bin_count=config.hist_bin_count,
                    hist_shared_bins=config.hist_shared_bins,
                )
        finally:
            self._updating = False

    def _apply_preset(self, preset_id: str) -> None:
        preset = PRESET_BY_ID[preset_id]
        group = Y_GROUP_BY_ID[preset.y_group]
        self._set_controls_from_config(
            BinnedSummaryConfig(
                y_group=preset.y_group,
                source=group.sources[0],
                x_param=preset.x_param,
                glyph=preset.glyph,
                show_trend=preset.show_trend,
                show_hist=preset.show_hist,
                show_corr=preset.show_corr,
            )
        )
        self._sync_data_filter_for_metric()
        self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        self._refresh_generation += 1
        self._refresh_timer.start()

    def _start_refresh(self) -> None:
        gen = self._refresh_generation
        config = self._read_config()
        session_data = self._session_data()
        self.setWindowTitle(config.title)
        self._schedule_render(gen, config, session_data)

    def _schedule_render(
        self,
        gen: int,
        config: BinnedSummaryConfig,
        session_data: dict[str, dict],
    ) -> None:
        figsize = tuple(float(v) for v in self.figure.get_size_inches())
        base_dir = self._base_dir

        def render_fn() -> tuple[int, Figure | None]:
            fig = new_headless_figure(figsize)
            try:
                render_binned_summary(fig, config, session_data, base_dir)
            except Exception:
                import logging
                logging.getLogger(__name__).exception("Binned summary render failed")
                return gen, None
            return gen, fig

        self._render_task.schedule(render_fn)

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

    def _metric_filter_state(self) -> tuple[bool, bool]:
        y_group_id = (
            self._metric_panel.selected_id()
            if self._metric_panel is not None
            else None
        )
        group = Y_GROUP_BY_ID.get(y_group_id) if y_group_id else None
        if group is None:
            return False, False
        supports_filter = group.supports_data_filter
        has_beam_state = (
            supports_filter
            and data_source_is_timeslice(self._current_source())  # type: ignore[arg-type]
        )
        return supports_filter, has_beam_state

    def _sync_data_filter_for_metric(self) -> None:
        supports_filter, has_beam_state = self._metric_filter_state()
        sync_data_filter_panel(
            self._filter_panel,
            supports_filter=supports_filter,
            has_beam_state=has_beam_state,
            reset_defaults=True,
        )

    def _on_metric_selection_changed(self) -> None:
        if self._updating:
            return
        self._sync_data_filter_for_metric()
        self._refresh_x_combo()
        self._on_controls_changed()

    def _on_controls_changed(self, *_args) -> None:
        if self._updating:
            return
        self._sync_x_bins_control()
        self._sync_plot_style_controls()
        self._schedule_refresh()


def run_binned_summary_window(
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
        lambda: BinnedSummaryWindow(
            session_ids,
            base_dir,
            settings=settings,
            initial_preset=initial_preset,
        ),
    )
