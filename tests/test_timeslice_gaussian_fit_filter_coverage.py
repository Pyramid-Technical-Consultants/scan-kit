"""Tests for Gaussian fit filter spot coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scan_kit.common.timeslice_gaussian_fit_filter_coverage import (
    _combined_ic_spots_and_metrics,
    compute_session_gaussian_fit_filter_coverage,
    resolve_timeslice_gaussian_fit_filter_coverage_source,
)

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"


def test_g3_session_computes_confidence_and_peak_coverage() -> None:
    coverage = compute_session_gaussian_fit_filter_coverage("1091134775", str(TEST_DATA))
    assert coverage is not None
    assert set(coverage.confidence.ics) == {"ic1", "ic2"}
    assert coverage.peak is not None
    assert set(coverage.peak.ics) == {"ic1", "ic2"}

    for sweep in (coverage.confidence, coverage.peak):
        for ic_cov in sweep.ics.values():
            assert ic_cov.total_spots > 0
            assert ic_cov.coverage_pct.shape == sweep.thresholds.shape
            assert ic_cov.coverage_pct[0] > 0
            assert np.all(np.diff(ic_cov.coverage_pct) <= 0)
            assert ic_cov.full_coverage_breakpoint is not None


def test_g2_session_has_confidence_without_peak() -> None:
    coverage = compute_session_gaussian_fit_filter_coverage("590658542", str(TEST_DATA))
    assert coverage is not None
    assert coverage.confidence.ics["ic1"].total_spots > 0
    assert coverage.peak is None


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
