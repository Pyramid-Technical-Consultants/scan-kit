"""Tests for universal binned summary helpers, loader, and Qt shell."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QSplitter

from scan_kit.common.plotting import (
    assign_bin_centers,
    build_bin_edges,
    collect_unique_bins,
    prepare_binned_column,
)
from scan_kit.views.binned_summary_catalog import (
    PRESET_DOSE_RATIO_ENERGY,
    PRESET_SIGMA_ENERGY,
    X_ENERGY,
    X_TARGET_MU,
    Y_DOSE_RATIO,
    Y_POSITION_ERROR,
    Y_SIGMA,
    BinnedSummaryConfig,
)
from scan_kit.views.binned_summary_data import (
    available_series_keys,
    available_x_params,
    available_y_groups,
    default_config,
    load_session_summary_table,
    load_sessions_summary,
)
from scan_kit.views.binned_summary_ui import render_binned_summary
from scan_kit.views.binned_summary_window import BinnedSummaryWindow

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"
G3_SESSION = "1091134775"
G2_SESSION = "590658542"


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


def test_binned_summary_window_smoke() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = BinnedSummaryWindow([G3_SESSION], str(TEST_DATA))
    assert window._session_data
    assert isinstance(window.centralWidget(), QSplitter)
    window._apply_preset(PRESET_DOSE_RATIO_ENERGY)
    window._apply_preset(PRESET_SIGMA_ENERGY)
    # Switch X to target MU when available.
    if X_TARGET_MU in window._x_avail:
        idx = window._x_combo.findData(X_TARGET_MU)
        window._x_combo.setCurrentIndex(idx)
    window.close()
    del app


def test_position_error_series_when_available() -> None:
    session_data = load_sessions_summary([G3_SESSION], str(TEST_DATA))
    if Y_POSITION_ERROR not in available_y_groups(session_data):
        pytest.skip("position error unavailable")
    keys = available_series_keys(session_data, Y_POSITION_ERROR)
    assert "ic1_x_err" in keys
