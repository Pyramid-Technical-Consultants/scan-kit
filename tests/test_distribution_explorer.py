"""Tests for Distribution Explorer loaders, renderer, and Qt shell."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from PySide6.QtWidgets import QApplication

from scan_kit.common.settings import ViewSettings
from scan_kit.views.distribution_catalog import (
    MODE_CONFIDENCE_TIMESLICE,
    MODE_GAUSSIAN_FILTER,
    MODE_POSITION_ERROR_SPOT,
    MODE_POSITION_ERROR_TIMESLICE,
    MODE_POSITION_SPOT,
    MODE_POSITION_TIMESLICE,
    MODE_SIGMA_ERROR_TIMESLICE,
    MODE_SIGMA_SPOT,
    MODE_SIGMA_TIMESLICE,
    PRESETS,
    DistributionConfig,
)
from scan_kit.views.distribution_data import (
    default_mode,
    load_sessions_for_mode,
    mode_has_data,
)
from scan_kit.views.distribution_ui import render_distribution
from scan_kit.views.distribution_window import DistributionExplorerWindow

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"
G3_SESSION = "1091134775"
G2_SESSION = "590658542"


@pytest.mark.parametrize("mode", [preset.mode for preset in PRESETS])
def test_mode_has_data_g3(mode: str) -> None:
    if mode == MODE_GAUSSIAN_FILTER:
        pytest.skip("Gaussian filter coverage requires G3 fit columns")
    assert mode_has_data(mode, [G3_SESSION], str(TEST_DATA))


def test_default_mode_picks_first_available() -> None:
    mode = default_mode([G3_SESSION], str(TEST_DATA))
    assert mode in {
        MODE_POSITION_SPOT,
        MODE_POSITION_ERROR_SPOT,
        MODE_SIGMA_SPOT,
        MODE_POSITION_TIMESLICE,
        MODE_POSITION_ERROR_TIMESLICE,
        MODE_SIGMA_TIMESLICE,
        MODE_SIGMA_ERROR_TIMESLICE,
        MODE_CONFIDENCE_TIMESLICE,
    }


@pytest.mark.parametrize("mode", [preset.mode for preset in PRESETS])
def test_render_distribution_headless(mode: str) -> None:
    session_data = load_sessions_for_mode(mode, [G3_SESSION], str(TEST_DATA))
    if not session_data:
        pytest.skip(f"{mode} unavailable in fixture")
    fig = plt.figure()
    render_distribution(
        fig,
        DistributionConfig(mode=mode),
        session_data,
        str(TEST_DATA),
    )
    assert fig.axes
    plt.close(fig)


@pytest.mark.parametrize(
    ("mode", "plot_style"),
    [
        (MODE_POSITION_SPOT, "scatter"),
        (MODE_POSITION_ERROR_SPOT, "contour"),
        (MODE_SIGMA_TIMESLICE, "scatter"),
    ],
)
def test_render_distribution_plot_styles(mode: str, plot_style: str) -> None:
    if not mode_has_data(mode, [G3_SESSION], str(TEST_DATA)):
        pytest.skip(f"{mode} unavailable in fixture")
    session_data = load_sessions_for_mode(mode, [G3_SESSION], str(TEST_DATA))
    fig = plt.figure()
    render_distribution(
        fig,
        DistributionConfig(mode=mode, plot_style=plot_style),  # type: ignore[arg-type]
        session_data,
        str(TEST_DATA),
    )
    assert fig.axes
    plt.close(fig)


def test_distribution_explorer_window_smoke() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = DistributionExplorerWindow([G3_SESSION], str(TEST_DATA))
    window._apply_preset(MODE_SIGMA_TIMESLICE)
    window._start_refresh()
    for _ in range(30):
        app.processEvents()
    window._apply_preset(MODE_POSITION_ERROR_TIMESLICE)
    window._start_refresh()
    for _ in range(30):
        app.processEvents()
    window._apply_preset(MODE_POSITION_SPOT)
    window._start_refresh()
    for _ in range(30):
        app.processEvents()
    window._plot_style_segmented.set_current("scatter")
    window._on_plot_style_segment_changed("scatter")
    window._start_refresh()
    for _ in range(30):
        app.processEvents()
    window._bg_segmented.set_current("on")
    window._on_bg_segment_changed("on")
    window._start_refresh()
    for _ in range(30):
        app.processEvents()
    window.close()
    del app


def test_spot_sigma_loads_for_g3() -> None:
    data = load_sessions_for_mode(
        MODE_SIGMA_SPOT,
        [G3_SESSION],
        str(TEST_DATA),
    )
    if not mode_has_data(MODE_SIGMA_SPOT, [G3_SESSION], str(TEST_DATA)):
        pytest.skip("spot sigma unavailable in fixture")
    assert data


def test_spot_position_loads_for_g3() -> None:
    if not mode_has_data(MODE_POSITION_SPOT, [G3_SESSION], str(TEST_DATA)):
        pytest.skip("spot position unavailable in fixture")
    data = load_sessions_for_mode(MODE_POSITION_SPOT, [G3_SESSION], str(TEST_DATA))
    assert data
    sample = next(iter(data.values()))
    assert sample.plan_x is not None
    assert sample.plan_y is not None


def test_spot_position_error_loads_for_g3() -> None:
    data = load_sessions_for_mode(
        MODE_POSITION_ERROR_SPOT,
        [G3_SESSION],
        str(TEST_DATA),
    )
    assert data


def test_timeslice_position_loads_for_g3() -> None:
    if not mode_has_data(MODE_POSITION_TIMESLICE, [G3_SESSION], str(TEST_DATA)):
        pytest.skip("timeslice position unavailable in fixture")
    data = load_sessions_for_mode(
        MODE_POSITION_TIMESLICE,
        [G3_SESSION],
        str(TEST_DATA),
    )
    assert data


def test_sigma_error_timeslice_loads_for_g3() -> None:
    if not mode_has_data(MODE_SIGMA_ERROR_TIMESLICE, [G3_SESSION], str(TEST_DATA)):
        pytest.skip("sigma error unavailable in fixture")
    data = load_sessions_for_mode(
        MODE_SIGMA_ERROR_TIMESLICE,
        [G3_SESSION],
        str(TEST_DATA),
    )
    assert data


def test_gaussian_filter_mode_g3_when_available() -> None:
    if not mode_has_data(MODE_GAUSSIAN_FILTER, [G3_SESSION], str(TEST_DATA)):
        pytest.skip("Gaussian filter coverage unavailable in fixture")
    data = load_sessions_for_mode(
        MODE_GAUSSIAN_FILTER,
        [G3_SESSION],
        str(TEST_DATA),
        settings=ViewSettings(),
    )
    assert data
