"""Tests for shared unified-view data filters."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from scan_kit.common.data_filter import (
    FILTER_ALL,
    FILTER_BEAM_OFF,
    FILTER_BEAM_ON,
    FILTER_BEAM_BOTH,
    FILTER_LOWER_95,
    FILTER_MAD_OUTLIERS,
    FILTER_UPPER_95,
    DataFilterSelection,
    beam_state_mask,
    default_beam_state_filter,
    filter_binned_session_data,
    filter_mask_from_columns,
    filter_session_dict,
    filter_session_ic_xy,
    filter_session_position_errors,
    modified_z,
)
from scan_kit.common.session_ic_xy import SessionIcXYData
from scan_kit.common.timeslice_position_error import SessionPositionErrors
from tests.conftest import G3_SESSION, HAS_TEST_DATA, TEST_DATA


def test_modified_z_is_zero_for_constant_values() -> None:
    z = modified_z(np.full(10, 2.0))
    assert np.allclose(z, 0.0)


def test_filter_mask_from_columns_upper_tail() -> None:
    session = {
        "energy": np.array([100.0] * 5),
        "ic1_x_err": np.array([0.0, 0.1, 0.2, 0.3, 10.0]),
        "ic1_y_err": np.array([0.0, 0.1, 0.2, 0.3, 10.0]),
        "ic2_x_err": np.array([0.0, 0.1, 0.2, 0.3, 10.0]),
        "ic2_y_err": np.array([0.0, 0.1, 0.2, 0.3, 10.0]),
    }
    mask = filter_mask_from_columns(
        session,
        ("ic1_x_err", "ic1_y_err", "ic2_x_err", "ic2_y_err"),
        FILTER_UPPER_95,
        FILTER_BEAM_BOTH,
    )
    assert mask.sum() == 1
    assert mask[-1]


def test_combined_domain_and_beam_filters() -> None:
    session = {
        "energy": np.array([100.0] * 4),
        "ic1_current": np.array([10.0, 20.0, 1.0, 2.0]),
        "ic2_current": np.array([10.0, 20.0, 1.0, 2.0]),
        "beam_on": np.array([True, True, False, False]),
    }
    mask = filter_mask_from_columns(
        session,
        ("ic1_current", "ic2_current"),
        FILTER_ALL,
        FILTER_BEAM_ON,
    )
    assert mask.tolist() == [True, True, False, False]

    out = filter_binned_session_data(
        {"s1": session},
        ("ic1_current", "ic2_current"),
        DataFilterSelection(FILTER_ALL, FILTER_BEAM_ON),
    )
    kept = out["s1"]["ic1_current"]
    assert np.sum(np.isfinite(kept)) == 2
    assert kept[0] == 10.0 and kept[1] == 20.0


def test_filter_session_position_errors_mad_outliers() -> None:
    ic1_x = np.zeros(100)
    ic1_x[-1] = 100.0
    data = SessionPositionErrors(
        ic1_x=ic1_x,
        ic1_y=np.zeros_like(ic1_x),
        ic2_x=np.zeros_like(ic1_x),
        ic2_y=np.zeros_like(ic1_x),
    )
    filtered = filter_session_position_errors(
        data, DataFilterSelection(FILTER_MAD_OUTLIERS, FILTER_BEAM_BOTH),
    )
    assert np.isfinite(filtered.ic1_x[-1])
    assert np.sum(np.isfinite(filtered.ic1_x)) == 1


def test_filter_binned_session_data_position_error() -> None:
    session = {
        "session_id": "s1",
        "energy": np.array([100.0, 100.0, 100.0, 100.0, 100.0]),
        "ic1_x_err": np.array([0.0, 0.1, 0.2, 0.3, 5.0]),
        "ic1_y_err": np.array([0.0, 0.1, 0.2, 0.3, 0.0]),
        "ic2_x_err": np.array([0.0, 0.1, 0.2, 0.3, 0.0]),
        "ic2_y_err": np.array([0.0, 0.1, 0.2, 0.3, 0.0]),
    }
    out = filter_binned_session_data(
        {"s1": session},
        ("ic1_x_err", "ic1_y_err", "ic2_x_err", "ic2_y_err"),
        DataFilterSelection(FILTER_UPPER_95, FILTER_BEAM_BOTH),
    )
    kept = out["s1"]["ic1_x_err"]
    assert len(kept) == 1
    assert kept[0] == 5.0


def test_filter_all_leaves_session_unchanged() -> None:
    session = {
        "session_id": "s1",
        "energy": np.array([1.0, 2.0]),
        "ic1_x_err": np.array([0.0, 5.0]),
        "ic1_y_err": np.array([0.0, 0.0]),
        "ic2_x_err": np.array([0.0, 0.0]),
        "ic2_y_err": np.array([0.0, 0.0]),
    }
    out = filter_binned_session_data(
        {"s1": session},
        ("ic1_x_err", "ic1_y_err", "ic2_x_err", "ic2_y_err"),
        DataFilterSelection(FILTER_ALL, FILTER_BEAM_BOTH),
    )
    assert np.array_equal(out["s1"]["ic1_x_err"], session["ic1_x_err"])


def test_beam_state_mask_on_off_both() -> None:
    beam_on = np.array([True, True, False, False])
    assert beam_state_mask(beam_on, FILTER_BEAM_ON, 4).tolist() == [True, True, False, False]
    assert beam_state_mask(beam_on, FILTER_BEAM_OFF, 4).tolist() == [False, False, True, True]
    assert beam_state_mask(beam_on, FILTER_BEAM_BOTH, 4).all()


def test_beam_state_mask_length_mismatch_still_filters() -> None:
    beam_on = np.array([True, False, True])
    assert beam_state_mask(beam_on, FILTER_BEAM_ON, 4).tolist() == [True, False, True, True]
    assert beam_state_mask(beam_on, FILTER_BEAM_OFF, 4).tolist() == [False, True, False, False]
    long_on = np.array([True, False, True, False, True])
    assert beam_state_mask(long_on, FILTER_BEAM_ON, 4).tolist() == [True, False, True, False]


def test_filter_session_position_errors_beam_off() -> None:
    beam_on = np.array([True, False, True, False])
    data = SessionPositionErrors(
        ic1_x=np.array([1.0, 2.0, 3.0, 4.0]),
        ic1_y=np.zeros(4),
        ic2_x=np.zeros(4),
        ic2_y=np.zeros(4),
        beam_on=beam_on,
    )
    filtered = filter_session_position_errors(
        data, DataFilterSelection(FILTER_ALL, FILTER_BEAM_OFF),
    )
    assert np.isfinite(filtered.ic1_x).tolist() == [False, True, False, True]


def test_default_beam_filter_is_on_when_beam_state_exists() -> None:
    assert default_beam_state_filter(has_beam_state=True) == FILTER_BEAM_ON
    assert default_beam_state_filter(has_beam_state=False) == FILTER_BEAM_BOTH


def test_filter_session_dict_subsets_all_1d_arrays() -> None:
    session = {
        "session_id": "s1",
        "t": np.arange(4.0),
        "ic1": np.array([1.0, 2.0, 3.0, 4.0]),
        "beam_on": np.array([True, False, True, False]),
    }
    out = filter_session_dict(session, np.array([True, False, True, False]))
    assert out["session_id"] == "s1"
    assert out["t"].tolist() == [0.0, 2.0]
    assert out["ic1"].tolist() == [1.0, 3.0]
    assert out["beam_on"].tolist() == [True, True]
    assert len(out["beam_on"]) == len(out["ic1"])


@pytest.mark.skipif(not HAS_TEST_DATA, reason="test_data/ is not available")
def test_g3_beam_on_filter_drops_real_samples() -> None:
    from scan_kit.common import detect_beam_on_mask
    from scan_kit.common.timeslice_table import load_session_timeslice_frames

    opened = load_session_timeslice_frames(G3_SESSION, str(TEST_DATA), max_frames=1)
    if opened is None:
        pytest.skip(f"no timeslice frames for {G3_SESSION}")
    beam_on = detect_beam_on_mask(opened[1][0])
    if beam_on is None:
        pytest.skip("G3 frame has no beam_on mask")
    n_on = int(np.asarray(beam_on).sum())
    n = len(beam_on)
    assert 0 < n_on < n
    session = {
        "ic1": np.ones(n),
        "beam_on": np.asarray(beam_on, dtype=bool),
    }
    both = filter_mask_from_columns(session, ("ic1",), FILTER_ALL, FILTER_BEAM_BOTH)
    on = filter_mask_from_columns(session, ("ic1",), FILTER_ALL, FILTER_BEAM_ON)
    assert int(on.sum()) == n_on
    assert int(both.sum()) == n
    filtered = filter_session_dict(session, on)
    assert len(filtered["ic1"]) == n_on
    assert len(filtered["beam_on"]) == n_on
    assert np.all(filtered["beam_on"])


def test_filter_session_ic_xy_all_nan_severity_is_silent() -> None:
    n = 4
    data = SessionIcXYData(
        ic1_x=np.full(n, np.nan),
        ic1_y=np.full(n, np.nan),
        ic2_x=np.full(n, np.nan),
        ic2_y=np.full(n, np.nan),
        plan_x=np.zeros(n),
        plan_y=np.zeros(n),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        filtered = filter_session_ic_xy(data, FILTER_LOWER_95)
    assert not any("All-NaN slice" in str(w.message) for w in caught)
    assert not np.isfinite(filtered.ic1_x).any()
