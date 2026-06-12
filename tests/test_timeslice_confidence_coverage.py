"""Tests for spot-level confidence threshold coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scan_kit.common.timeslice_confidence_coverage import (
    compute_session_confidence_coverage,
    resolve_timeslice_confidence_coverage_source,
)

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"


def test_g3_session_computes_coverage_curves() -> None:
    coverage = compute_session_confidence_coverage("1091134775", str(TEST_DATA))
    assert coverage is not None
    assert set(coverage.axes) == {"ic1", "ic2"}
    for axis_cov in coverage.axes.values():
        assert axis_cov.total_spots > 0
        assert axis_cov.coverage_pct.shape == coverage.thresholds.shape
        assert axis_cov.coverage_pct[0] > 0
        assert np.all(np.diff(axis_cov.coverage_pct) <= 0)
        assert axis_cov.full_coverage_breakpoint is not None
        assert 0 <= axis_cov.full_coverage_breakpoint <= 100


def test_g2_session_with_confidence_computes_coverage() -> None:
    coverage = compute_session_confidence_coverage("590658542", str(TEST_DATA))
    assert coverage is not None
    assert coverage.axes["ic1"].total_spots > 0


def test_resolve_source_requires_confidence_columns() -> None:
    assert resolve_timeslice_confidence_coverage_source(["layer_id", "spot_no"]) is None
