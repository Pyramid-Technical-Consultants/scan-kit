"""Tests for unified timeslice replay loader, config, and Qt shell."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scan_kit.common.schema import C_MAG_FIELD_X, C_MAG_FIELD_Y, resolve_concept_column
from scan_kit.views.timeslice_replay_channels import (
    PRESET_CHANNELS,
    PRESET_FIELD,
    PRESET_IC_CURRENT,
    PRESET_SIGMA,
    available_channel_keys,
    build_replay_config,
    default_selected_keys,
    filter_available_keys,
)
from scan_kit.views.timeslice_replay_ui import render_timeslice_replay
from scan_kit.views.timeslice_replay_window import TimesliceReplayWindow
from tests.conftest import G2_SESSION, G3_SESSION, TEST_DATA


def test_g3_field_column_aliases() -> None:
    cols = pd.read_csv(
        TEST_DATA / "1091134775" / "1091134775" / "layer-71" / "run-0"
        / "timeslice_data_device_units.csv",
        nrows=0,
    ).columns
    assert resolve_concept_column(cols, C_MAG_FIELD_X) == "r_tx2_probe_x"
    assert resolve_concept_column(cols, C_MAG_FIELD_Y) == "r_tx2_probe_y"


def test_g2_field_column_aliases() -> None:
    cols = pd.read_csv(
        TEST_DATA / "883144654" / "layer-9" / "run-0"
        / "timeslice_data_device_units.csv",
        nrows=0,
    ).columns
    assert resolve_concept_column(cols, C_MAG_FIELD_X) == "field_c_x"
    assert resolve_concept_column(cols, C_MAG_FIELD_Y) == "field_c_y"


def test_load_catalog_includes_ic_and_field_g3(g3_timeline_catalog) -> None:
    data = g3_timeline_catalog
    assert data
    assert data["n_samples"] > 0
    assert "ic1" in data
    assert "bx" in data
    assert "b_mag" in data
    assert np.nanmax(np.abs(data["bx"])) > 0


def test_load_catalog_includes_sigma_g2(g2_timeline_catalog) -> None:
    data = g2_timeline_catalog
    assert data
    assert "sigma_ic1_x" in data
    assert np.isfinite(data["sigma_ic1_x"]).any()
    assert np.nanmedian(data["sigma_ic1_x"]) > 0.5


def test_load_catalog_includes_sigma_g3(g3_timeline_catalog) -> None:
    data = g3_timeline_catalog
    assert data
    assert np.isfinite(data["sigma_ic1_x"]).any()
    assert np.all(data["sigma_ic1_x"][np.isfinite(data["sigma_ic1_x"])] > 0)


def test_load_catalog_includes_ddose_when_dose_present(g3_timeline_catalog) -> None:
    data = g3_timeline_catalog
    assert data
    if data.get("has_ddose"):
        assert "ic1_ddose" in data
        assert len(data["ic1_ddose"]) == data["n_samples"]


def test_available_and_default_selection(g3_timeslice_catalog) -> None:
    available = available_channel_keys(g3_timeslice_catalog)
    assert "ic1" in available or "bx" in available
    selected = default_selected_keys(available)
    assert selected
    assert set(selected) <= available


def test_build_replay_config_field_preset(g3_timeslice_catalog) -> None:
    available = available_channel_keys(g3_timeslice_catalog)
    keys = filter_available_keys(("bx", "by"), available)
    assert keys == ["bx", "by"]
    config = build_replay_config(keys, g3_timeslice_catalog)
    assert [t.key for t in config.traces] == ["bx", "by"]
    assert config.timeline_key == "b_mag"
    assert config.scatter.mode == "single"


def test_build_replay_config_sigma_preset(g2_timeslice_catalog) -> None:
    available = available_channel_keys(g2_timeslice_catalog)
    keys = filter_available_keys(PRESET_CHANNELS[PRESET_SIGMA], available)
    config = build_replay_config(keys, g2_timeslice_catalog)
    assert config.scatter.mode == "per_trace"
    assert config.timeline_key == "sigma_ic1_x"


def test_build_replay_config_mixed_hides_scatter(g3_timeslice_catalog) -> None:
    available = available_channel_keys(g3_timeslice_catalog)
    keys = filter_available_keys(("ic1", "bx"), available)
    if len(keys) < 2:
        pytest.skip("session missing mixed channel families")
    config = build_replay_config(keys, g3_timeslice_catalog)
    assert config.scatter.mode == "none"


def test_render_timeslice_replay_headless(g3_timeslice_catalog) -> None:
    import matplotlib.pyplot as plt

    available = available_channel_keys(g3_timeslice_catalog)
    keys = default_selected_keys(available)
    config = build_replay_config(keys, g3_timeslice_catalog)
    fig = plt.figure()
    render_timeslice_replay(fig, config, g3_timeslice_catalog, str(TEST_DATA))
    assert fig.axes
    plt.close(fig)


@pytest.mark.slow
def test_timeslice_replay_window_smoke(qapp) -> None:
    from PySide6.QtWidgets import QSplitter

    window = TimesliceReplayWindow([G3_SESSION], str(TEST_DATA))
    assert window._session_data
    assert isinstance(window.centralWidget(), QSplitter)
    assert window.side_panel is not None
    keys = window._selected_keys()
    assert keys
    window._apply_preset(PRESET_FIELD)
    field_keys = window._selected_keys()
    assert set(field_keys) <= {"bx", "by"}
    window._apply_preset(PRESET_IC_CURRENT)
    window.close()


def test_plot_view_shell_side_panel_resizable(qapp) -> None:
    from PySide6.QtWidgets import QLabel

    from scan_kit.views.plot_view_shell import PlotViewWindow, make_side_panel_column

    window = PlotViewWindow(title="Shell Test")
    panel, layout = make_side_panel_column()
    layout.addWidget(QLabel("Controls"))
    window.set_side_panel(panel)
    sizes = window._splitter.sizes()
    assert len(sizes) == 2
    assert sizes[1] >= window._side_min_width
    window._splitter.setSizes([800, 400])
    assert window._splitter.sizes()[1] >= 300
    window.close()


def test_resolve_frame_energy_empty_frame_uses_frame_index() -> None:
    from scan_kit.views.timeslice_replay_common import resolve_frame_energy

    empty = pd.DataFrame({"_layer_idx": pd.Series([], dtype=int)})
    assert resolve_frame_energy(
        empty,
        3,
        energy_by_layer=None,
        energy_by_idx={3: 150.0},
        layer_col="layer_id",
    ) == 150.0
