"""Tests for Gaussian fit filter spot coverage."""



from __future__ import annotations



from pathlib import Path



import numpy as np
import pytest

from scan_kit.common.timeslice_gaussian_fit_filter_coverage import (
    PLOTTED_ERROR_CONFIDENCE_ISSUE_CODES,
    SPOT_ERROR_CODE_NAMES,
    SPOT_ERROR_CODES,
    _build_orphan_spot_errors,
    _build_weighted_position_rms_sweep,
    _combine_ic_error_confidence_counts,
    _combined_ic_spots_and_metrics,
    _confidence_invalid,
    _confidence_peak_weight,
    _orphan_ic_spot_set,
    _weighted_axis_position_error,
    compute_session_gaussian_fit_filter_coverage,
    resolve_timeslice_gaussian_fit_filter_coverage_source,
    spot_error_code_name,
)



TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"





def test_g3_session_computes_confidence_and_peak_coverage() -> None:

    coverage = compute_session_gaussian_fit_filter_coverage("1091134775", str(TEST_DATA))

    assert coverage is not None

    assert set(coverage.confidence.ics) == {"ic1", "ic2"}

    assert coverage.peak is not None

    assert set(coverage.peak.ics) == {"ic1", "ic2"}

    assert set(coverage.error_confidence_issues.ics) == {"ic1", "ic2"}

    assert set(coverage.orphan_spot_errors.ics) == {"ic1", "ic2"}
    assert coverage.weighted_position_rms is not None
    assert set(coverage.weighted_position_rms.ics) == {"ic1", "ic2"}

    assert len(coverage.error_confidence_issues.codes) == len(SPOT_ERROR_CODES)



    for sweep in (coverage.confidence, coverage.peak):

        for ic_cov in sweep.ics.values():

            assert ic_cov.total_spots > 0

            assert ic_cov.coverage_pct.shape == sweep.thresholds.shape

            assert ic_cov.coverage_pct[0] > 0

            assert np.all(np.diff(ic_cov.coverage_pct) <= 0)

            assert ic_cov.full_coverage_breakpoint is not None



    issues = coverage.error_confidence_issues.ics["ic1"].counts_by_code

    assert issues.shape == SPOT_ERROR_CODES.shape

    assert issues.sum() > 0

    assert issues[0] > 0





def test_g2_session_has_confidence_without_peak() -> None:

    coverage = compute_session_gaussian_fit_filter_coverage("590658542", str(TEST_DATA))

    assert coverage is not None

    assert coverage.confidence.ics["ic1"].total_spots > 0

    assert coverage.peak is None





def test_session_with_spot_position_ok_alias_computes_coverage() -> None:

    """Older G3 exports use r_ic*_spot_position_ok instead of ic*_fit_ok."""

    coverage = compute_session_gaussian_fit_filter_coverage("1262268206", str(TEST_DATA))

    assert coverage is not None

    assert coverage.peak is not None

    for sweep in (coverage.confidence, coverage.peak):

        for ic_cov in sweep.ics.values():

            assert ic_cov.total_spots > 0

            assert ic_cov.coverage_pct[0] > 0

            assert ic_cov.full_coverage_breakpoint is not None



    issues = coverage.error_confidence_issues.ics["ic1"].counts_by_code

    assert issues[1] > 0

    assert issues[0] == 0

    plotted = issues[PLOTTED_ERROR_CONFIDENCE_ISSUE_CODES]

    assert plotted.sum() > 0



    orphans = coverage.orphan_spot_errors.ics["ic2"]

    assert orphans.orphan_spots == 1

    assert orphans.counts_by_code[1] > 0

    assert coverage.weighted_position_rms is not None
    assert coverage.weighted_position_rms.ics["ic2"].total_spots > 0

    peak_series = coverage.orphan_spot_peaks.ics["ic2"]

    assert len(peak_series) == 1

    spot = peak_series[0]

    assert spot.spot_no == 3282.0

    assert len(spot.peak_x) == 8

    assert np.all(spot.error_x == 1)

    assert spot.error_y.tolist().count(0) == 1

    assert float(np.nanmax(spot.peak_x)) == pytest.approx(0.908348, rel=1e-5)


def test_plotted_error_confidence_issue_codes_skip_routine_codes() -> None:

    assert PLOTTED_ERROR_CONFIDENCE_ISSUE_CODES.tolist() == [2, 3, 4, 5]





def test_spot_error_code_names_match_g3_enum() -> None:

    assert spot_error_code_name(0) == "No error"

    assert spot_error_code_name(1) == "Peak below threshold"

    assert spot_error_code_name(5) == "Solution invalid"

    assert len(SPOT_ERROR_CODE_NAMES) == 6





def test_confidence_invalid_flags_negative_and_nan() -> None:
    vals = np.array([-1.0, 0.0, np.nan, 80.0])
    invalid = _confidence_invalid(vals)
    assert invalid.tolist() == [True, False, True, False]


def test_confidence_peak_weight_uses_peak_when_available() -> None:
    assert _confidence_peak_weight(80.0, 2.0) == 160.0
    assert _confidence_peak_weight(80.0, np.nan) == 80.0


def test_weighted_axis_position_error_respects_threshold() -> None:
    samples = [
        (1.0, 0.0, 50.0, 1.0),
        (3.0, 0.0, 90.0, 2.0),
    ]
    assert _weighted_axis_position_error(samples, 0.0) == pytest.approx(590.0 / 230.0)
    assert _weighted_axis_position_error(samples, 80.0) == pytest.approx(3.0)
    assert _weighted_axis_position_error(samples, 95.0) is None


def test_build_weighted_position_rms_decreases_with_higher_threshold() -> None:
    spot_samples = {
        "ic1_x": {
            (1.0, 10.0): [(0.0, 0.0, 90.0, 1.0), (0.2, 0.0, 60.0, 1.0)],
            (2.0, 20.0): [(0.0, 0.0, 90.0, 1.0), (1.0, 0.0, 60.0, 1.0)],
        },
        "ic1_y": {
            (1.0, 10.0): [(0.0, 0.0, 90.0, 1.0), (0.0, 0.0, 60.0, 1.0)],
            (2.0, 20.0): [(0.0, 0.0, 90.0, 1.0), (0.0, 0.0, 60.0, 1.0)],
        },
        "ic2_x": {},
        "ic2_y": {},
    }
    sweep = _build_weighted_position_rms_sweep(
        spot_samples, np.array([0.0, 80.0], dtype=float)
    )
    assert sweep is not None
    ic1 = sweep.ics["ic1"]
    assert ic1.spots_used[0] == 2
    assert ic1.spots_used[1] == 2
    assert ic1.rms_xy_mm[0] > ic1.rms_xy_mm[1]





def test_orphan_ic_spot_set_matches_zero_threshold_coverage() -> None:

    all_spots = {

        "ic1_x": {(1.0, 10.0), (2.0, 20.0)},

        "ic1_y": {(1.0, 10.0), (2.0, 20.0)},

        "ic2_x": set(),

        "ic2_y": set(),

    }

    max_metric = {

        "ic1_x": {(1.0, 10.0): 90.0},

        "ic1_y": {(1.0, 10.0): 80.0},

        "ic2_x": {},

        "ic2_y": {},

    }

    orphans = _orphan_ic_spot_set(all_spots, max_metric, "ic1")

    assert orphans == {(2.0, 20.0)}





def test_build_orphan_spot_errors_keeps_all_row_codes() -> None:

    summary = _build_orphan_spot_errors(

        {"ic1": {(2.0, 20.0)}},

        {"ic1": np.array([0, 2, 0, 1, 0, 0], dtype=int), "ic2": np.zeros(6, dtype=int)},

    )

    ic1 = summary.ics["ic1"]

    assert ic1.orphan_spots == 1

    assert ic1.counts_by_code[1] == 2

    assert ic1.counts_by_code[3] == 1





def test_ic_error_confidence_counts_sum_x_and_y_axes() -> None:

    axis_counts = {

        "ic1_x": np.array([0, 2, 0, 0, 0, 0]),

        "ic1_y": np.array([1, 0, 0, 0, 0, 0]),

        "ic2_x": np.zeros(6, dtype=int),

        "ic2_y": np.zeros(6, dtype=int),

    }

    combined = _combine_ic_error_confidence_counts(axis_counts, "ic1")

    assert combined.tolist() == [1, 2, 0, 0, 0, 0]





def test_combined_ic_coverage_is_stricter_than_single_axis() -> None:

    """Combined X/Y breakpoint is at most the weaker axis's peak metric."""

    all_spots = {

        "ic1_x": {(1.0, 10.0)},

        "ic1_y": {(1.0, 10.0)},

        "ic2_x": set(),

        "ic2_y": set(),

    }

    max_metric = {

        "ic1_x": {(1.0, 10.0): 90.0},

        "ic1_y": {(1.0, 10.0): 70.0},

        "ic2_x": {},

        "ic2_y": {},

    }

    _spots, combined = _combined_ic_spots_and_metrics(all_spots, max_metric, "ic1")

    assert combined[(1.0, 10.0)] == 70.0





def test_resolve_source_requires_confidence_columns() -> None:

    assert resolve_timeslice_gaussian_fit_filter_coverage_source(["layer_id", "spot_no"]) is None


