"""Equivalence between Distribution and Binned adapters for shared sources."""

from __future__ import annotations

import numpy as np
import pytest

from scan_kit.data import (
    DATA_SOURCE_SPOT_ISO,
    DATA_SOURCE_TIMESLICE_ISO,
    LoadOptions,
    REFERENCE_ISO,
    SessionContext,
    load,
)
from scan_kit.data.adapters.binned import ic12_to_binned_columns
from scan_kit.data.adapters.distribution import ic12_to_session_xy
from scan_kit.data.sources.ic12_pos_diff import SOURCE_IC12_POS_DIFF
from scan_kit.views.binned_summary_data import (
    load_session_summary_table,
    load_session_timeslice_summary_table,
)
from scan_kit.views.distribution_catalog import (
    MODE_IC12_POS_DIFF_SPOT,
    MODE_IC12_POS_DIFF_TIMESLICE,
    MODE_POSITION_ERROR_SPOT,
    MODE_POSITION_ERROR_TIMESLICE,
    MODE_SIGMA_SPOT,
    MODE_SIGMA_TIMESLICE,
)
from scan_kit.views.distribution_data import load_sessions_for_mode
from tests.conftest import G2_SESSION, G3_SESSION, TEST_DATA


@pytest.mark.parametrize("session_id", [G3_SESSION, G2_SESSION])
def test_ic12_spot_distribution_matches_binned(session_id: str) -> None:
    dist = load_sessions_for_mode(
        MODE_IC12_POS_DIFF_SPOT, [session_id], str(TEST_DATA),
    )
    binned = load_session_summary_table(
        session_id, str(TEST_DATA), data_source=DATA_SOURCE_SPOT_ISO,
    )
    if session_id not in dist or binned is None or "ic12_x_diff" not in binned:
        pytest.skip("IC12 spot data unavailable")
    xy = dist[session_id]
    np.testing.assert_allclose(xy.ic1_x, binned["ic12_x_diff"], equal_nan=True)
    np.testing.assert_allclose(xy.ic1_y, binned["ic12_y_diff"], equal_nan=True)


@pytest.mark.slow
@pytest.mark.parametrize("session_id", [G3_SESSION, G2_SESSION])
def test_ic12_timeslice_distribution_matches_binned(session_id: str) -> None:
    dist = load_sessions_for_mode(
        MODE_IC12_POS_DIFF_TIMESLICE, [session_id], str(TEST_DATA),
    )
    binned = load_session_timeslice_summary_table(
        session_id, str(TEST_DATA), data_source=DATA_SOURCE_SPOT_ISO,
    )
    if session_id not in dist or binned is None or "ic12_x_diff" not in binned:
        pytest.skip("IC12 timeslice data unavailable")
    xy = dist[session_id]
    np.testing.assert_allclose(xy.ic1_x, binned["ic12_x_diff"], equal_nan=True)
    np.testing.assert_allclose(xy.ic1_y, binned["ic12_y_diff"], equal_nan=True)


@pytest.mark.parametrize("session_id", [G3_SESSION])
def test_ic12_adapters_agree_on_registry_payload(session_id: str) -> None:
    ctx = SessionContext(session_id, str(TEST_DATA))
    payload = load(
        SOURCE_IC12_POS_DIFF,
        ctx,
        LoadOptions(data_source=DATA_SOURCE_SPOT_ISO),
    )
    if payload is None:
        pytest.skip("IC12 spot payload unavailable")
    xy = ic12_to_session_xy(payload)
    cols = ic12_to_binned_columns(payload)
    assert xy is not None and cols is not None
    np.testing.assert_allclose(xy.ic1_x, cols["ic12_x_diff"], equal_nan=True)
    np.testing.assert_allclose(xy.ic1_y, cols["ic12_y_diff"], equal_nan=True)


@pytest.mark.slow
@pytest.mark.parametrize("session_id", [G3_SESSION])
def test_ic12_timeslice_adapters_agree_on_registry_payload(session_id: str) -> None:
    ctx = SessionContext(session_id, str(TEST_DATA))
    payload = load(
        SOURCE_IC12_POS_DIFF,
        ctx,
        LoadOptions(
            granularity=GRANULARITY_TIMESLICE_SAMPLE,
            data_source=DATA_SOURCE_TIMESLICE_ISO,
        ),
    )
    if payload is None:
        pytest.skip("IC12 timeslice payload unavailable")
    xy = ic12_to_session_xy(payload)
    cols = ic12_to_binned_columns(payload)
    assert xy is not None and cols is not None
    np.testing.assert_allclose(xy.ic1_x, cols["ic12_x_diff"], equal_nan=True)
    np.testing.assert_allclose(xy.ic1_y, cols["ic12_y_diff"], equal_nan=True)


@pytest.mark.parametrize("session_id", [G3_SESSION, G2_SESSION])
def test_position_error_spot_distribution_matches_binned(session_id: str) -> None:
    dist = load_sessions_for_mode(
        MODE_POSITION_ERROR_SPOT, [session_id], str(TEST_DATA),
    )
    binned = load_session_summary_table(session_id, str(TEST_DATA))
    if session_id not in dist or binned is None or "ic1_x_err" not in binned:
        pytest.skip("position error spot data unavailable")
    errors = dist[session_id]
    np.testing.assert_allclose(errors.ic1_x, binned["ic1_x_err"], equal_nan=True)
    np.testing.assert_allclose(errors.ic2_y, binned["ic2_y_err"], equal_nan=True)


@pytest.mark.parametrize("session_id", [G3_SESSION, G2_SESSION])
def test_sigma_spot_distribution_matches_binned(session_id: str) -> None:
    dist = load_sessions_for_mode(
        MODE_SIGMA_SPOT, [session_id], str(TEST_DATA),
    )
    binned = load_session_summary_table(session_id, str(TEST_DATA))
    if session_id not in dist or binned is None or "ic1_sig_x" not in binned:
        pytest.skip("sigma spot data unavailable")
    sigmas = dist[session_id]
    np.testing.assert_allclose(sigmas.ic1_x, binned["ic1_sig_x"], equal_nan=True)
    np.testing.assert_allclose(sigmas.ic2_y, binned["ic2_sig_y"], equal_nan=True)


@pytest.mark.slow
@pytest.mark.parametrize("session_id", [G3_SESSION, G2_SESSION])
def test_position_error_timeslice_distribution_matches_binned(session_id: str) -> None:
    dist = load_sessions_for_mode(
        MODE_POSITION_ERROR_TIMESLICE, [session_id], str(TEST_DATA),
    )
    binned = load_session_timeslice_summary_table(session_id, str(TEST_DATA))
    if session_id not in dist or binned is None or "ic1_x_err" not in binned:
        pytest.skip("position error timeslice data unavailable")
    errors = dist[session_id]
    np.testing.assert_allclose(errors.ic1_x, binned["ic1_x_err"], equal_nan=True)


@pytest.mark.slow
@pytest.mark.parametrize("session_id", [G3_SESSION, G2_SESSION])
def test_sigma_timeslice_distribution_matches_binned(session_id: str) -> None:
    dist = load_sessions_for_mode(
        MODE_SIGMA_TIMESLICE, [session_id], str(TEST_DATA),
    )
    binned = load_session_timeslice_summary_table(session_id, str(TEST_DATA))
    if session_id not in dist or binned is None or "ic1_sig_x" not in binned:
        pytest.skip("sigma timeslice data unavailable")
    sigmas = dist[session_id]
    np.testing.assert_allclose(sigmas.ic1_x, binned["ic1_sig_x"], equal_nan=True)
