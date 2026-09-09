"""Tests for binned-summary interlock threshold overlays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scan_kit.common.interlock_thresholds import (
    DOSE_GATE_LEVELS,
    SIGMA_TOLERANCE_FRACTIONS,
    X_ENERGY,
    X_TARGET_MU,
    Y_DOSE_ERROR,
    Y_POSITION_ERROR,
    Y_SIGMA,
    Y_SIGMA_ERROR,
    dose_gate_threshold_pct,
    interlock_overlay_available,
    load_expected_sigmas_by_session,
    sigma_tolerance_band,
)
from scan_kit.views.binned_summary_catalog import BinnedSummaryConfig, GLYPH_BOX, GLYPH_VIOLIN
from scan_kit.views.binned_summary_ui import render_binned_summary
from scan_kit.views.binned_summary_data import load_sessions_summary

_ROOT = Path(__file__).resolve().parent.parent
_TEST_DATA = _ROOT / "test_data"
_SESSION = "1943968267"


def test_interlock_overlay_available_by_metric() -> None:
    assert interlock_overlay_available(Y_POSITION_ERROR, X_ENERGY)
    assert interlock_overlay_available(Y_POSITION_ERROR, "target_mu")
    assert interlock_overlay_available(Y_SIGMA, X_ENERGY, glyph=GLYPH_VIOLIN)
    assert interlock_overlay_available(Y_SIGMA_ERROR, X_ENERGY, glyph=GLYPH_VIOLIN)
    assert not interlock_overlay_available(Y_SIGMA, "target_mu")
    assert not interlock_overlay_available(Y_SIGMA_ERROR, "spot_time")
    assert not interlock_overlay_available(Y_SIGMA, X_ENERGY, glyph="scatter")
    assert not interlock_overlay_available("dose_error", X_ENERGY)
    assert interlock_overlay_available(Y_DOSE_ERROR, X_TARGET_MU)
    assert interlock_overlay_available(Y_DOSE_ERROR, X_TARGET_MU, glyph="scatter")
    assert not interlock_overlay_available(Y_DOSE_ERROR, X_ENERGY)


def test_dose_gate_threshold_pct_formula() -> None:
    assert dose_gate_threshold_pct(0.1, 0.01) == pytest.approx(3.0)
    assert dose_gate_threshold_pct(1.0, 0.02) == pytest.approx(2.2)


def test_sigma_tolerance_band_symmetric() -> None:
    lower, upper = sigma_tolerance_band(5.0, 0.20)
    assert lower == pytest.approx(4.0)
    assert upper == pytest.approx(6.0)


def test_load_expected_sigmas_by_session_fixture() -> None:
    energies = [70.0, 250.0]
    by_session = load_expected_sigmas_by_session(
        [_SESSION],
        energies,
        str(_TEST_DATA),
        series_key="ic1_sig_x",
    )
    assert _SESSION in by_session
    assert by_session[_SESSION][250.0] == pytest.approx(2.588, rel=1e-3)


def test_render_binned_summary_sigma_with_interlock_overlay() -> None:
    session_data = load_sessions_summary([_SESSION], str(_TEST_DATA))
    assert session_data
    config = BinnedSummaryConfig(
        y_group=Y_SIGMA,
        x_param=X_ENERGY,
        glyph=GLYPH_VIOLIN,
        show_trend=False,
        show_interlock_thresholds=True,
    )
    fig = plt.figure()
    render_binned_summary(fig, config, session_data, str(_TEST_DATA))
    assert fig.axes
    line_count = sum(len(ax.get_lines()) for ax in fig.axes)
    assert line_count >= len(SIGMA_TOLERANCE_FRACTIONS) * 2
    plt.close(fig)


def test_render_binned_summary_dose_error_mu_with_interlock_overlay() -> None:
    session_data = load_sessions_summary([_SESSION], str(_TEST_DATA))
    assert session_data
    config = BinnedSummaryConfig(
        y_group=Y_DOSE_ERROR,
        x_param=X_TARGET_MU,
        glyph=GLYPH_BOX,
        show_trend=False,
        show_interlock_thresholds=True,
        n_bins=5,
    )
    fig = plt.figure()
    render_binned_summary(fig, config, session_data, str(_TEST_DATA))
    assert fig.axes
    line_count = sum(len(ax.get_lines()) for ax in fig.axes)
    assert line_count >= len(DOSE_GATE_LEVELS) * 2
    plt.close(fig)
