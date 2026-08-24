"""Qt shell for the universal binned summary viewer."""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..common import ViewSettings
from .binned_summary_catalog import (
    DATA_SOURCE_SPOT,
    GLYPH_BOX,
    GLYPH_MEAN,
    GLYPH_VIOLIN,
    PRESETS,
    PRESET_BY_ID,
    VIEW_OPTIONS,
    X_ENERGY,
    X_PARAMS,
    BinnedSummaryConfig,
)
from .binned_summary_data import (
    available_x_params_for_source,
    default_config,
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
from .unified_catalog import option_key
from .unified_view_controls import DataSourceOptionPanel


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
        self._session_data_cache[DATA_SOURCE_SPOT] = self._spot_data
        self._option_availability = probe_view_option_availability(
            self._session_ids,
            self._base_dir,
            spot_data=self._spot_data,
        )
        self._x_avail = available_x_params_for_source(
            self._spot_data, DATA_SOURCE_SPOT,
        )
        self._updating = False
        self._metric_panel: DataSourceOptionPanel | None = None
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
            self._refresh_plot()

    def _build_controls(self) -> QWidget:
        panel, layout = make_side_panel_column()

        layout.addWidget(
            make_presets_menu_button(
                [
                    (
                        preset.id,
                        preset.label,
                        self._option_availability.get(
                            option_key(DATA_SOURCE_SPOT, preset.y_group),
                            False,
                        )
                        and preset.x_param in self._x_avail,
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

        glyph_group = QGroupBox("Glyph")
        glyph_layout = QVBoxLayout(glyph_group)
        self._glyph_group = QButtonGroup(self)
        self._glyph_box = QRadioButton("Box")
        self._glyph_violin = QRadioButton("Violin")
        self._glyph_mean = QRadioButton("Mean")
        self._glyph_box.setChecked(True)
        for btn, value in (
            (self._glyph_box, GLYPH_BOX),
            (self._glyph_violin, GLYPH_VIOLIN),
            (self._glyph_mean, GLYPH_MEAN),
        ):
            self._glyph_group.addButton(btn)
            btn.setProperty("glyph", value)
            btn.toggled.connect(self._on_controls_changed)
            glyph_layout.addWidget(btn)
        layout.addWidget(glyph_group)

        opts = QGroupBox("Options")
        opt_layout = QVBoxLayout(opts)
        self._trend_check = QCheckBox("Trend Line")
        self._trend_check.setChecked(True)
        self._trend_check.toggled.connect(self._on_controls_changed)
        opt_layout.addWidget(self._trend_check)

        self._hist_check = QCheckBox("Histogram Panel")
        self._hist_check.toggled.connect(self._on_controls_changed)
        opt_layout.addWidget(self._hist_check)

        self._corr_check = QCheckBox("Correlation Panel")
        self._corr_check.toggled.connect(self._on_controls_changed)
        opt_layout.addWidget(self._corr_check)

        self._fliers_check = QCheckBox("Show Box Outliers")
        self._fliers_check.toggled.connect(self._on_controls_changed)
        opt_layout.addWidget(self._fliers_check)
        layout.addWidget(opts)

        if not self._spot_data:
            note = QLabel("No summary data found for the selected sessions.")
            note.setWordWrap(True)
            layout.addWidget(note)

        layout.addStretch(1)
        return panel

    def _current_source(self) -> str:
        if self._metric_panel is None:
            return DATA_SOURCE_SPOT
        return self._metric_panel.selected_source()

    def _session_data(self) -> dict[str, dict]:
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
        for btn in (self._glyph_box, self._glyph_violin, self._glyph_mean):
            if btn.isChecked():
                return str(btn.property("glyph"))
        return GLYPH_BOX

    def _read_config(self) -> BinnedSummaryConfig:
        y_group = (
            self._metric_panel.selected_id()
            if self._metric_panel is not None
            else None
        )
        return BinnedSummaryConfig(
            y_group=y_group or PRESETS[0].y_group,
            source=self._current_source(),
            x_param=self._x_combo.currentData(),
            glyph=self._selected_glyph(),  # type: ignore[arg-type]
            show_trend=self._trend_check.isChecked(),
            show_hist=self._hist_check.isChecked(),
            show_corr=self._corr_check.isChecked(),
            show_fliers=self._fliers_check.isChecked(),
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
            {
                GLYPH_BOX: self._glyph_box,
                GLYPH_VIOLIN: self._glyph_violin,
                GLYPH_MEAN: self._glyph_mean,
            }.get(config.glyph, self._glyph_box).setChecked(True)
            self._trend_check.setChecked(config.show_trend)
            self._hist_check.setChecked(config.show_hist)
            self._corr_check.setChecked(config.show_corr)
            self._fliers_check.setChecked(config.show_fliers)
        finally:
            self._updating = False

    def _apply_preset(self, preset_id: str) -> None:
        preset = PRESET_BY_ID[preset_id]
        self._set_controls_from_config(
            BinnedSummaryConfig(
                y_group=preset.y_group,
                source=DATA_SOURCE_SPOT,
                x_param=preset.x_param,
                glyph=preset.glyph,
                show_trend=preset.show_trend,
                show_hist=preset.show_hist,
                show_corr=preset.show_corr,
            )
        )
        self._refresh_plot()

    def _on_metric_selection_changed(self) -> None:
        if self._updating:
            return
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
