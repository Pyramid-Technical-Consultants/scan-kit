"""Qt shell for the universal binned summary viewer."""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..common import ViewSettings
from ..common.data_filter import FILTER_ALL, FILTER_BEAM_BOTH, FILTER_BEAM_ON
from .binned_summary_catalog import (
    DATA_SOURCE_SPOT,
    DATA_SOURCE_TIMESLICE,
    GLYPH_VIOLIN,
    PRESETS,
    PRESET_BY_ID,
    VIEW_OPTIONS,
    X_ENERGY,
    X_PARAMS,
    Y_CURRENT_RATIO,
    Y_DOSE_RATE,
    Y_IC_CURRENT,
    Y_GROUP_BY_ID,
    BinnedSummaryConfig,
)
from .binned_summary_data import (
    available_x_params_for_source,
    default_config,
    load_sessions_current_ratios,
    load_sessions_dose_rate,
    load_sessions_ic_current,
    load_sessions_for_source,
    probe_view_option_availability,
)
from .binned_summary_ui import render_binned_summary
from .plot_view_shell import (
    PlotViewWindow,
    make_presets_menu_button,
    make_side_panel_column,
    run_view_window,
)
from .unified_catalog import BINNED_PLOT_STYLES, option_key
from .unified_view_controls import (
    DataFilterPanel,
    DataSourceOptionPanel,
    PlotStylePanel,
    sync_data_filter_panel,
)

_OPT_TREND = "trend"
_OPT_HIST = "hist"
_OPT_CORR = "corr"
_OPT_FLIERS = "fliers"


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
        self._session_data_cache: dict[str, dict[str, dict]] = {}
        self._spot_data = load_sessions_for_source(
            self._session_ids, self._base_dir, DATA_SOURCE_SPOT, settings=settings,
        )
        self._dose_rate_data = load_sessions_dose_rate(
            self._session_ids, self._base_dir,
        )
        self._current_ratio_data = load_sessions_current_ratios(
            self._session_ids, self._base_dir, settings=settings,
        )
        self._ic_current_data = load_sessions_ic_current(
            self._session_ids, self._base_dir, settings=settings,
        )
        self._session_data_cache[DATA_SOURCE_SPOT] = self._spot_data
        self._option_availability = probe_view_option_availability(
            self._session_ids,
            self._base_dir,
            spot_data=self._spot_data,
            dose_rate_data=self._dose_rate_data,
            current_ratio_data=self._current_ratio_data,
            ic_current_data=self._ic_current_data,
            settings=settings,
        )
        self._x_avail: set[str] = set()
        self._updating = False
        self._metric_panel: DataSourceOptionPanel | None = None
        self._plot_style_panel: PlotStylePanel | None = None
        self._filter_panel: DataFilterPanel | None = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(60)
        self._refresh_timer.timeout.connect(self._refresh_plot)

        self.set_side_panel(self._build_controls())
        if initial_preset and initial_preset in PRESET_BY_ID:
            self._apply_preset(initial_preset)
        else:
            cfg = default_config(
                self._spot_data,
                source=DATA_SOURCE_SPOT,
                option_availability=self._option_availability,
            )
            self._set_controls_from_config(cfg)
            self._sync_data_filter_for_metric()
            self._refresh_plot()

    def _build_controls(self) -> QWidget:
        panel, layout = make_side_panel_column()

        layout.addWidget(
            make_presets_menu_button(
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
        )

        self._metric_panel = DataSourceOptionPanel(
            on_selection_changed=self._on_metric_selection_changed,
        )
        self._metric_panel.configure(
            VIEW_OPTIONS,
            self._option_availability,
            group_title="Y Metric",
            preferred_source=DATA_SOURCE_SPOT,
        )
        layout.addWidget(self._metric_panel)

        x_group = QGroupBox("X Parameter")
        x_layout = QVBoxLayout(x_group)
        self._x_combo = QComboBox()
        self._x_model = QStandardItemModel(self._x_combo)
        self._x_combo.setModel(self._x_model)
        self._refresh_x_combo()
        self._x_combo.currentIndexChanged.connect(self._on_controls_changed)
        x_layout.addWidget(self._x_combo)
        layout.addWidget(x_group)

        self._plot_style_panel = PlotStylePanel(
            BINNED_PLOT_STYLES,
            on_selection_changed=self._on_controls_changed,
            current=GLYPH_VIOLIN,
        )
        self._plot_style_panel.add_checkbox(_OPT_TREND, "Trend Line", checked=True)
        self._plot_style_panel.add_checkbox(_OPT_HIST, "Histogram Panel")
        self._plot_style_panel.add_checkbox(_OPT_CORR, "Correlation Panel")
        self._plot_style_panel.add_checkbox(_OPT_FLIERS, "Show Box Outliers")
        layout.addWidget(self._plot_style_panel)

        self._filter_panel = DataFilterPanel(
            on_selection_changed=self._on_controls_changed,
            domain_current=FILTER_ALL,
            beam_current=FILTER_BEAM_ON,
        )
        layout.addWidget(self._filter_panel)

        if not (
            self._spot_data
            or self._dose_rate_data
            or self._current_ratio_data
            or self._ic_current_data
        ):
            note = QLabel("No summary data found for the selected sessions.")
            note.setWordWrap(True)
            layout.addWidget(note)

        layout.addStretch(1)
        return panel

    def _session_data_for_y_group(self, y_group: str) -> dict[str, dict]:
        if y_group == Y_DOSE_RATE:
            return self._dose_rate_data
        if y_group == Y_CURRENT_RATIO:
            return self._current_ratio_data
        if y_group == Y_IC_CURRENT:
            return self._ic_current_data
        group = Y_GROUP_BY_ID.get(y_group)
        if group is not None and group.sources[0] == DATA_SOURCE_TIMESLICE:
            cached = self._session_data_cache.get(DATA_SOURCE_TIMESLICE)
            if cached is not None:
                return cached
            loaded = load_sessions_for_source(
                self._session_ids,
                self._base_dir,
                DATA_SOURCE_TIMESLICE,
                settings=self._settings,
            )
            self._session_data_cache[DATA_SOURCE_TIMESLICE] = loaded
            return loaded
        return self._spot_data

    def _x_avail_for_y_group(self, y_group: str) -> set[str]:
        group = Y_GROUP_BY_ID.get(y_group)
        if group is None:
            return set()
        source = group.sources[0]
        session_data = self._session_data_for_y_group(y_group)
        return available_x_params_for_source(session_data, source)  # type: ignore[arg-type]

    def _preset_is_available(self, preset) -> bool:
        group = Y_GROUP_BY_ID[preset.y_group]
        if not self._option_availability.get(
            option_key(group.sources[0], preset.y_group),
            False,
        ):
            return False
        return preset.x_param in self._x_avail_for_y_group(preset.y_group)

    def _current_source(self) -> str:
        if self._metric_panel is None:
            return DATA_SOURCE_SPOT
        return self._metric_panel.selected_source()

    def _session_data(self) -> dict[str, dict]:
        y_group = (
            self._metric_panel.selected_id()
            if self._metric_panel is not None
            else None
        )
        if y_group == Y_DOSE_RATE:
            return self._dose_rate_data
        if y_group == Y_CURRENT_RATIO:
            return self._current_ratio_data
        if y_group == Y_IC_CURRENT:
            return self._ic_current_data
        source = self._current_source()
        cached = self._session_data_cache.get(source)
        if cached is not None:
            return cached
        loaded = load_sessions_for_source(
            self._session_ids,
            self._base_dir,
            source,  # type: ignore[arg-type]
            settings=self._settings,
        )
        self._session_data_cache[source] = loaded
        return loaded

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
        return BinnedSummaryConfig(
            y_group=y_group or PRESETS[0].y_group,
            source=self._current_source(),
            x_param=self._x_combo.currentData(),
            glyph=self._selected_glyph(),  # type: ignore[arg-type]
            show_trend=panel.is_checked(_OPT_TREND) if panel else True,
            show_hist=panel.is_checked(_OPT_HIST) if panel else False,
            show_corr=panel.is_checked(_OPT_CORR) if panel else False,
            show_fliers=panel.is_checked(_OPT_FLIERS) if panel else False,
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
            if self._plot_style_panel is not None:
                self._plot_style_panel.set_current(config.glyph)
                self._plot_style_panel.set_checked(_OPT_TREND, config.show_trend)
                self._plot_style_panel.set_checked(_OPT_HIST, config.show_hist)
                self._plot_style_panel.set_checked(_OPT_CORR, config.show_corr)
                self._plot_style_panel.set_checked(_OPT_FLIERS, config.show_fliers)
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
        self._refresh_plot()

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
            and self._current_source() == DATA_SOURCE_TIMESLICE
        )
        return supports_filter, has_beam_state

    def _sync_data_filter_for_metric(self) -> None:
        supports_filter, has_beam_state = self._metric_filter_state()
        sync_data_filter_panel(
            self._filter_panel,
            supports_filter=supports_filter,
            has_beam_state=has_beam_state,
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
        self._refresh_timer.start()

    def _refresh_plot(self) -> None:
        config = self._read_config()
        session_data = self._session_data()
        self.setWindowTitle(config.title)
        render_binned_summary(
            self.figure, config, session_data, self._base_dir,
        )
        self.draw_idle()


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
