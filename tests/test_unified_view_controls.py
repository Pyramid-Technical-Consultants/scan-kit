"""Tests for shared unified-view catalog helpers and controls."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from scan_kit.views.unified_catalog import (
    DATA_SOURCE_SPOT,
    DATA_SOURCE_TIMESLICE,
    UnifiedViewOption,
    default_option_id,
    default_source,
    options_for_source,
    source_has_available_options,
)
from scan_kit.views.unified_view_controls import (
    DataSourceOptionPanel,
    PlotStylePanel,
)
from scan_kit.views.unified_catalog import BINNED_PLOT_STYLES, PLOT_STYLE_BOX

OPTIONS = (
    UnifiedViewOption("spot_a", "Spot A", DATA_SOURCE_SPOT),
    UnifiedViewOption("spot_b", "Spot B", DATA_SOURCE_SPOT),
    UnifiedViewOption("ts_a", "Timeslice A", DATA_SOURCE_TIMESLICE),
    UnifiedViewOption("ts_b", "Timeslice B", DATA_SOURCE_TIMESLICE),
)


def test_options_for_source_filters_by_kind() -> None:
    spot = options_for_source(OPTIONS, DATA_SOURCE_SPOT)
    assert [opt.id for opt in spot] == ["spot_a", "spot_b"]


def test_default_source_prefers_first_with_availability() -> None:
    availability = {
        "spot:spot_a": False,
        "spot:spot_b": False,
        "timeslice:ts_a": True,
        "timeslice:ts_b": False,
    }
    assert default_source(OPTIONS, availability) == DATA_SOURCE_TIMESLICE
    assert default_option_id(OPTIONS, availability) == "ts_a"


def test_default_option_respects_explicit_source() -> None:
    availability = {
        "spot:spot_a": False,
        "spot:spot_b": True,
        "timeslice:ts_a": True,
        "timeslice:ts_b": False,
    }
    assert default_option_id(OPTIONS, availability, source=DATA_SOURCE_SPOT) == "spot_b"


def test_source_has_available_options() -> None:
    availability = {
        "spot:spot_a": True,
        "spot:spot_b": False,
        "timeslice:ts_a": False,
        "timeslice:ts_b": False,
    }
    assert source_has_available_options(OPTIONS, availability, DATA_SOURCE_SPOT)
    assert not source_has_available_options(OPTIONS, availability, DATA_SOURCE_TIMESLICE)


def test_data_source_option_panel_spot_only_hides_source_group(qapp) -> None:
    panel = DataSourceOptionPanel()
    availability = {
        "spot:spot_a": True,
        "spot:spot_b": True,
        "timeslice:ts_a": False,
        "timeslice:ts_b": False,
    }
    panel.configure(OPTIONS, availability, group_title="Metric")
    assert not panel._source_segmented.isVisible()
    assert panel.selected_id() == "spot_a"


def test_data_source_option_panel_switch_source_filters_list(qapp: QApplication) -> None:
    panel = DataSourceOptionPanel()
    availability = {
        "spot:spot_a": False,
        "spot:spot_b": True,
        "timeslice:ts_a": True,
        "timeslice:ts_b": False,
    }
    panel.configure(OPTIONS, availability)
    assert panel.selected_id() == "spot_b"
    panel._set_source(DATA_SOURCE_TIMESLICE, refresh_list=True)
    assert panel.selected_id() == "ts_a"
    assert panel._option_list.count() == 2


def test_data_source_option_panel_preserves_metric_on_source_switch(
    qapp: QApplication,
) -> None:
    panel = DataSourceOptionPanel()
    options = (
        UnifiedViewOption("sigma", "Sigma", DATA_SOURCE_SPOT),
        UnifiedViewOption("sigma", "Sigma", DATA_SOURCE_TIMESLICE),
        UnifiedViewOption("dose_error", "Dose Error", DATA_SOURCE_SPOT),
    )
    availability = {
        "spot:sigma": True,
        "timeslice:sigma": True,
        "spot:dose_error": True,
    }
    panel.configure(options, availability, group_title="Y Metric")
    panel.select_id("sigma", source=DATA_SOURCE_SPOT)
    panel._source_segmented.set_current(DATA_SOURCE_TIMESLICE)
    panel._on_source_segment_changed(DATA_SOURCE_TIMESLICE)
    assert panel.selected_source() == DATA_SOURCE_TIMESLICE
    assert panel.selected_id() == "sigma"


def test_data_source_option_panel_source_switch_falls_back_when_unavailable(
    qapp: QApplication,
) -> None:
    panel = DataSourceOptionPanel()
    options = (
        UnifiedViewOption("dose_error", "Dose Error", DATA_SOURCE_SPOT),
        UnifiedViewOption("sigma", "Sigma", DATA_SOURCE_TIMESLICE),
    )
    availability = {
        "spot:dose_error": True,
        "timeslice:sigma": True,
    }
    panel.configure(options, availability, group_title="Y Metric")
    panel.select_id("dose_error", source=DATA_SOURCE_SPOT)
    panel._source_segmented.set_current(DATA_SOURCE_TIMESLICE)
    panel._on_source_segment_changed(DATA_SOURCE_TIMESLICE)
    assert panel.selected_id() == "sigma"


def test_data_source_option_panel_composite_and_legacy_availability(
    qapp: QApplication,
) -> None:
    panel = DataSourceOptionPanel()
    availability = {"position_error_timeslice": True}
    options = (
        UnifiedViewOption("position_error_timeslice", "Position Error", DATA_SOURCE_TIMESLICE),
    )
    panel.configure(options, availability, group_title="Distribution")
    assert panel.selected_id() == "position_error_timeslice"


def test_plot_style_panel_selects_glyph(qapp: QApplication) -> None:
    panel = PlotStylePanel(BINNED_PLOT_STYLES, current=PLOT_STYLE_BOX)
    assert panel.selected_key() == PLOT_STYLE_BOX
    panel.set_current("violin")
    assert panel.selected_key() == "violin"


def test_plot_style_panel_style_options(qapp: QApplication) -> None:
    panel = PlotStylePanel(BINNED_PLOT_STYLES, current=PLOT_STYLE_BOX)
    panel.add_checkbox("trend", "Trend Line", checked=True)
    panel.add_percent_spinbox("cutoff", "Contour Cutoff", value=30)
    assert panel.is_checked("trend")
    panel.set_checked("trend", False)
    assert not panel.is_checked("trend")
    assert panel.spin_value("cutoff") == 30
    panel.set_spin_value("cutoff", 20)
    assert panel.spin_value("cutoff") == 20


def test_data_filter_panel_selects_filters(qapp: QApplication) -> None:
    from scan_kit.common.data_filter import FILTER_ALL, FILTER_BEAM_ON, FILTER_MAD_OUTLIERS
    from scan_kit.views.unified_view_controls import DataFilterPanel

    panel = DataFilterPanel(domain_current=FILTER_ALL, beam_current=FILTER_BEAM_ON)
    assert panel.selected_domain() == FILTER_ALL
    assert panel.selected_beam_state() == FILTER_BEAM_ON
    panel.set_domain(FILTER_MAD_OUTLIERS)
    panel.set_beam_state(FILTER_BEAM_ON)
    assert panel.selected_domain() == FILTER_MAD_OUTLIERS
    assert panel.selection().beam_state_filter == FILTER_BEAM_ON
