"""Tests for universal binned summary helpers, loader, and Qt shell."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PySide6.QtWidgets import QSplitter

from scan_kit.common import DEFAULT_SESSION_COLORS
from scan_kit.common.plotting import (
    assign_bin_centers,
    build_bin_edges,
    collect_unique_bins,
    finish_view,
    prepare_binned_column,
)
from scan_kit.common.settings import ViewSettings
from scan_kit.views.binned_summary_catalog import (
    PRESET_BY_ID,
    PRESET_DOSE_RATIO_ENERGY,
    PRESET_SIGMA_ENERGY,
    PRESETS,
    X_ENERGY,
    X_TARGET_MU,
    Y_DOSE_RATIO,
    Y_DOSE_RATE,
    Y_CURRENT_RATIO,
    Y_POSITION_ERROR,
    Y_SIGMA,
    BinnedSummaryConfig,
)
from scan_kit.views.binned_summary_data import (
    available_series_keys,
    available_x_params,
    available_x_params_for_source,
    available_y_groups,
    default_config,
    load_session_summary_table,
    load_sessions_dose_rate,
    load_sessions_current_ratios,
    load_session_timeslice_summary_table,
    load_sessions_summary,
    probe_view_option_availability,
)
from scan_kit.views.unified_catalog import DATA_SOURCE_SPOT, DATA_SOURCE_TIMESLICE
from scan_kit.views.binned_summary_ui import render_binned_summary
from scan_kit.views.binned_summary_window import BinnedSummaryWindow

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"
G3_SESSION = "1091134775"
G2_SESSION = "590658542"


def _config_for_preset(preset_id: str) -> BinnedSummaryConfig:
    preset = PRESET_BY_ID[preset_id]
    return BinnedSummaryConfig(
        y_group=preset.y_group,
        x_param=preset.x_param,
        glyph=preset.glyph,
        show_trend=preset.show_trend,
        show_hist=preset.show_hist,
        show_corr=preset.show_corr,
    )


def _run_preset_matplotlib(
    session_ids: list[str],
    base_dir: str,
    preset_id: str,
    *,
    settings: ViewSettings | None = None,
) -> None:
    session_data = load_sessions_summary(
        session_ids,
        base_dir,
        settings=settings,
    )
    if not session_data:
        return

    config = _config_for_preset(preset_id)
    fig = plt.figure(figsize=(16, 9))
    render_binned_summary(fig, config, session_data, base_dir)
    loaded_ids = list(session_data.keys())
    finish_view(
        fig,
        config.title,
        loaded_ids,
        DEFAULT_SESSION_COLORS[: len(loaded_ids)],
        base_dir=base_dir,
    )


def test_build_bin_edges_quantile() -> None:
    values = np.linspace(0, 100, 201)
    edges = build_bin_edges(values, mode="quantile", n_bins=4)
    assert edges.size >= 2
    assert edges[0] <= values.min()
    assert edges[-1] >= values.max()


def test_assign_bin_centers_covers_range() -> None:
    edges = np.asarray([0.0, 10.0, 20.0, 30.0])
    centers = assign_bin_centers([5.0, 15.0, 25.0, np.nan], edges)
    assert centers[0] == pytest.approx(5.0)
    assert centers[1] == pytest.approx(15.0)
    assert centers[2] == pytest.approx(25.0)
    assert np.isnan(centers[3])


def test_prepare_binned_column_unique_and_quantile() -> None:
    session_data = {
        "a": {"energy": np.array([70.0, 100.0, 70.0]), "y": np.array([1.0, 2.0, 3.0])},
        "b": {"energy": np.array([100.0, 130.0]), "y": np.array([4.0, 5.0])},
    }
    prepared, cats = prepare_binned_column(session_data, "energy", mode="unique")
    assert cats == [70.0, 100.0, 130.0]
    assert np.all(prepared["a"]["_bin"] == prepared["a"]["energy"])

    continuous = {
        "a": {"target_mu": np.linspace(1, 10, 50), "y": np.ones(50)},
    }
    _prep2, cats2 = prepare_binned_column(
        continuous, "target_mu", mode="quantile", n_bins=5,
    )
    assert len(cats2) >= 2


def test_collect_unique_bins() -> None:
    session_data = {
        "a": {"energy": np.array([1.0, 2.0, np.nan])},
        "b": {"energy": np.array([2.0, 3.0])},
    }
    assert collect_unique_bins(session_data, "energy") == [1.0, 2.0, 3.0]


def test_load_summary_table_g3() -> None:
    data = load_session_summary_table(G3_SESSION, str(TEST_DATA))
    assert data is not None
    assert "energy" in data
    assert len(data["energy"]) > 0
    # At least one first-wave metric family should be present.
    assert any(
        key in data
        for key in (
            "ic21_ratio",
            "ic1_dose_err_pct",
            "ic1_x_err",
            "ic1_sig_x",
            "spot_time",
        )
    )


def test_load_summary_availability() -> None:
    session_data = load_sessions_summary([G3_SESSION, G2_SESSION], str(TEST_DATA))
    assert session_data
    y_avail = available_y_groups(session_data)
    x_avail = available_x_params(session_data)
    assert X_ENERGY in x_avail
    assert y_avail
    cfg = default_config(session_data)
    assert cfg.y_group in y_avail
    assert cfg.x_param in x_avail
    assert cfg.glyph == "violin"
    series = available_series_keys(session_data, cfg.y_group)
    assert series


def test_render_binned_summary_headless() -> None:
    session_data = load_sessions_summary([G3_SESSION], str(TEST_DATA))
    config = BinnedSummaryConfig(
        y_group=Y_DOSE_RATIO if Y_DOSE_RATIO in available_y_groups(session_data) else Y_SIGMA,
        x_param=X_ENERGY,
        glyph="box",
        show_trend=True,
    )
    if config.y_group == Y_SIGMA:
        config.glyph = "violin"
        config.show_trend = False
    fig = plt.figure()
    render_binned_summary(fig, config, session_data, str(TEST_DATA))
    assert fig.axes
    plt.close(fig)


def test_render_binned_summary_dose_rate_headless() -> None:
    session_data = load_sessions_dose_rate([G3_SESSION], str(TEST_DATA))
    if not session_data:
        pytest.skip("dose rate unavailable in fixture")
    config = BinnedSummaryConfig(
        y_group=Y_DOSE_RATE,
        x_param=X_ENERGY,
        glyph="mean",
        show_trend=True,
    )
    fig = plt.figure()
    render_binned_summary(fig, config, session_data, str(TEST_DATA))
    assert fig.axes
    plt.close(fig)


def test_render_binned_summary_current_ratio_headless() -> None:
    session_data = load_sessions_current_ratios([G3_SESSION], str(TEST_DATA))
    if not session_data:
        pytest.skip("current ratio data unavailable in fixture")
    config = BinnedSummaryConfig(
        y_group=Y_CURRENT_RATIO,
        x_param=X_ENERGY,
        glyph="mean",
        show_trend=True,
    )
    fig = plt.figure()
    render_binned_summary(fig, config, session_data, str(TEST_DATA))
    assert fig.axes
    plt.close(fig)


def test_render_binned_summary_quantile_x() -> None:
    session_data = load_sessions_summary([G3_SESSION], str(TEST_DATA))
    if X_TARGET_MU not in available_x_params(session_data):
        pytest.skip("target_mu unavailable in fixture")
    y = Y_DOSE_RATIO if Y_DOSE_RATIO in available_y_groups(session_data) else next(
        iter(available_y_groups(session_data))
    )
    config = BinnedSummaryConfig(y_group=y, x_param=X_TARGET_MU, glyph="box")
    fig = plt.figure()
    render_binned_summary(fig, config, session_data, str(TEST_DATA))
    assert fig.axes
    plt.close(fig)


def test_binned_summary_window_smoke(qapp) -> None:
    window = BinnedSummaryWindow([G3_SESSION], str(TEST_DATA))
    assert window._spot_data
    assert window._session_data()
    assert isinstance(window.centralWidget(), QSplitter)
    window._apply_preset(PRESET_DOSE_RATIO_ENERGY)
    window._apply_preset(PRESET_SIGMA_ENERGY)
    if X_TARGET_MU in window._x_avail:
        idx = window._x_combo.findData(X_TARGET_MU)
        window._x_combo.setCurrentIndex(idx)
    window.close()


def test_timeslice_summary_table_when_available() -> None:
    data = load_session_timeslice_summary_table(G3_SESSION, str(TEST_DATA))
    if data is None:
        pytest.skip("timeslice summary unavailable in fixture")
    assert "energy" in data
    assert any(key in data for key in ("ic1_x_err", "ic1_sig_x"))


def test_probe_view_option_availability_includes_timeslice() -> None:
    spot_data = load_sessions_summary([G3_SESSION], str(TEST_DATA))
    availability = probe_view_option_availability(
        [G3_SESSION], str(TEST_DATA), spot_data=spot_data,
    )
    assert any(key.startswith(f"{DATA_SOURCE_SPOT}:") for key in availability)
    if load_session_timeslice_summary_table(G3_SESSION, str(TEST_DATA)) is not None:
        assert any(key.startswith(f"{DATA_SOURCE_TIMESLICE}:") for key in availability)


def test_position_error_series_when_available() -> None:
    session_data = load_sessions_summary([G3_SESSION], str(TEST_DATA))
    if Y_POSITION_ERROR not in available_y_groups(session_data):
        pytest.skip("position error unavailable")
    keys = available_series_keys(session_data, Y_POSITION_ERROR)
    assert "ic1_x_err" in keys


@pytest.mark.parametrize("preset_id", [preset.id for preset in PRESETS])
def test_run_preset_matplotlib_all_presets(preset_id: str) -> None:
    _run_preset_matplotlib([G3_SESSION], str(TEST_DATA), preset_id)
    assert plt.get_fignums()
    plt.close("all")
