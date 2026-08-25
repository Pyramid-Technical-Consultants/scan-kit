"""Tests for Distribution Explorer loaders, renderer, and Qt shell."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from scan_kit.views.unified_catalog import REFERENCE_CHAMBER, REFERENCE_ISO
from scan_kit.common.settings import ViewSettings
from scan_kit.common.session_ic_xy import SessionIcXYData, ic12_position_diff
from scan_kit.views.distribution_catalog import (
    MODE_CONFIDENCE_TIMESLICE,
    MODE_GAUSSIAN_FILTER,
    MODE_IC12_POS_DIFF_SPOT,
    MODE_IC12_POS_DIFF_TIMESLICE,
    MODE_POSITION_ERROR_SPOT,
    MODE_POSITION_ERROR_TIMESLICE,
    MODE_POSITION_SPOT,
    MODE_POSITION_TIMESLICE,
    MODE_SIGMA_ERROR_SPOT,
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
from tests.conftest import G3_SESSION, TEST_DATA


def test_mode_availability_g3(g3_distribution_availability) -> None:
    availability = g3_distribution_availability
    assert any(availability.values())
    assert availability[MODE_POSITION_ERROR_SPOT]
    assert availability[MODE_SIGMA_SPOT]
    assert not availability.get(MODE_SIGMA_ERROR_SPOT, False)


def test_default_mode_picks_first_available(g3_distribution_availability) -> None:
    mode = default_mode(
        [G3_SESSION],
        str(TEST_DATA),
        availability=g3_distribution_availability,
    )
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


@pytest.mark.parametrize(
    "mode",
    [
        MODE_POSITION_ERROR_SPOT,
        MODE_SIGMA_TIMESLICE,
        MODE_POSITION_SPOT,
    ],
)
def test_render_distribution_headless_core(
    mode: str,
    g3_distribution_availability,
) -> None:
    if not g3_distribution_availability.get(mode, False):
        pytest.skip(f"{mode} unavailable in fixture")
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


@pytest.mark.slow
@pytest.mark.parametrize("mode", [preset.mode for preset in PRESETS])
def test_render_distribution_headless(mode: str, g3_distribution_availability) -> None:
    if mode == MODE_SIGMA_ERROR_SPOT:
        pytest.skip("spot sigma error requires per-spot sigma targets in fixture")
    if not g3_distribution_availability.get(mode, False):
        pytest.skip(f"{mode} unavailable in fixture")
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
def test_render_distribution_plot_styles(
    mode: str,
    plot_style: str,
    g3_distribution_availability,
) -> None:
    if not g3_distribution_availability.get(mode, False):
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


@pytest.mark.slow
def test_distribution_explorer_window_smoke(qapp, qt_wait) -> None:
    window = DistributionExplorerWindow([G3_SESSION], str(TEST_DATA))
    window._apply_preset(MODE_SIGMA_TIMESLICE)
    window._start_refresh()
    qt_wait(lambda: bool(window.figure.axes), timeout_ms=8000)
    window._apply_preset(MODE_POSITION_ERROR_TIMESLICE)
    window._start_refresh()
    qt_wait(lambda: bool(window.figure.axes), timeout_ms=8000)
    window._apply_preset(MODE_POSITION_SPOT)
    window._start_refresh()
    qt_wait(lambda: bool(window.figure.axes), timeout_ms=8000)
    window._plot_style_panel.set_current("scatter")
    window._schedule_refresh()
    qt_wait(lambda: window._plot_style_panel.selected_key() == "scatter")
    window._plot_style_panel.set_spin_value("contour_cutoff", 15)
    window._schedule_refresh()
    qt_wait(lambda: window._plot_style_panel.spin_value("contour_cutoff") == 15)
    window.close()


def test_spot_sigma_loads_for_g3(g3_distribution_availability) -> None:
    if not g3_distribution_availability.get(MODE_SIGMA_SPOT, False):
        pytest.skip("spot sigma unavailable in fixture")
    data = load_sessions_for_mode(
        MODE_SIGMA_SPOT,
        [G3_SESSION],
        str(TEST_DATA),
    )
    assert data


def test_spot_position_loads_for_g3(g3_distribution_availability) -> None:
    if not g3_distribution_availability.get(MODE_POSITION_SPOT, False):
        pytest.skip("spot position unavailable in fixture")
    data = load_sessions_for_mode(MODE_POSITION_SPOT, [G3_SESSION], str(TEST_DATA))
    assert data
    sample = next(iter(data.values()))
    assert sample.plan_x is not None
    assert sample.plan_y is not None


def test_spot_position_error_loads_for_g3(g3_distribution_availability) -> None:
    if not g3_distribution_availability.get(MODE_POSITION_ERROR_SPOT, False):
        pytest.skip("spot position error unavailable in fixture")
    data = load_sessions_for_mode(
        MODE_POSITION_ERROR_SPOT,
        [G3_SESSION],
        str(TEST_DATA),
    )
    assert data


def test_timeslice_position_loads_for_g3(g3_distribution_availability) -> None:
    if not g3_distribution_availability.get(MODE_POSITION_TIMESLICE, False):
        pytest.skip("timeslice position unavailable in fixture")
    data = load_sessions_for_mode(
        MODE_POSITION_TIMESLICE,
        [G3_SESSION],
        str(TEST_DATA),
    )
    assert data


def test_sigma_error_timeslice_loads_for_g3(g3_distribution_availability) -> None:
    if not g3_distribution_availability.get(MODE_SIGMA_ERROR_TIMESLICE, False):
        pytest.skip("sigma error unavailable in fixture")
    data = load_sessions_for_mode(
        MODE_SIGMA_ERROR_TIMESLICE,
        [G3_SESSION],
        str(TEST_DATA),
    )
    assert data


def test_gaussian_filter_mode_g3_when_available(g3_distribution_availability) -> None:
    if not g3_distribution_availability.get(MODE_GAUSSIAN_FILTER, False):
        pytest.skip("Gaussian filter coverage unavailable in fixture")
    data = load_sessions_for_mode(
        MODE_GAUSSIAN_FILTER,
        [G3_SESSION],
        str(TEST_DATA),
        settings=ViewSettings(),
    )
    assert data


def test_ic12_position_diff_helper() -> None:
    import numpy as np

    data = SessionIcXYData(
        ic1_x=np.array([0.0, 1.0]),
        ic1_y=np.array([2.0, 3.0]),
        ic2_x=np.array([1.0, 4.0]),
        ic2_y=np.array([3.0, 1.0]),
    )
    diff = ic12_position_diff(data)
    assert np.allclose(diff.ic1_x, [1.0, 3.0])
    assert np.allclose(diff.ic1_y, [1.0, -2.0])


def test_ic12_pos_diff_spot_loads_for_g3(g3_distribution_availability) -> None:
    if not g3_distribution_availability.get(MODE_IC12_POS_DIFF_SPOT, False):
        pytest.skip("IC2-IC1 spot diff unavailable in fixture")
    data = load_sessions_for_mode(
        MODE_IC12_POS_DIFF_SPOT,
        [G3_SESSION],
        str(TEST_DATA),
    )
    assert data


def test_ic12_pos_diff_timeslice_loads_for_g3(g3_distribution_availability) -> None:
    if not g3_distribution_availability.get(MODE_IC12_POS_DIFF_TIMESLICE, False):
        pytest.skip("IC2-IC1 timeslice diff unavailable in fixture")
    data = load_sessions_for_mode(
        MODE_IC12_POS_DIFF_TIMESLICE,
        [G3_SESSION],
        str(TEST_DATA),
    )
    assert data


def test_reference_frame_affects_spot_sigma_cache(g3_distribution_availability) -> None:
    if not g3_distribution_availability.get(MODE_SIGMA_SPOT, False):
        pytest.skip("spot sigma unavailable in fixture")
    iso = load_sessions_for_mode(
        MODE_SIGMA_SPOT,
        [G3_SESSION],
        str(TEST_DATA),
        reference_frame=REFERENCE_ISO,
    )
    chamber = load_sessions_for_mode(
        MODE_SIGMA_SPOT,
        [G3_SESSION],
        str(TEST_DATA),
        reference_frame=REFERENCE_CHAMBER,
    )
    assert iso and chamber


def test_render_ic12_pos_diff_headless(g3_distribution_availability) -> None:
    if not g3_distribution_availability.get(MODE_IC12_POS_DIFF_SPOT, False):
        pytest.skip("IC2-IC1 spot diff unavailable in fixture")
    session_data = load_sessions_for_mode(
        MODE_IC12_POS_DIFF_SPOT,
        [G3_SESSION],
        str(TEST_DATA),
    )
    fig = plt.figure()
    render_distribution(
        fig,
        DistributionConfig(mode=MODE_IC12_POS_DIFF_SPOT),
        session_data,
        str(TEST_DATA),
    )
    assert fig.axes
    plt.close(fig)
