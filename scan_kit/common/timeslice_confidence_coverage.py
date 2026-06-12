"""Backward-compatible re-exports for confidence-only Gaussian fit filter coverage."""

from __future__ import annotations

from .timeslice_gaussian_fit_filter_coverage import (
    CONFIDENCE_THRESHOLDS,
    THRESHOLD_STEP,
    TIMESLICE_CONFIDENCE_COVERAGE_COLS,
    AxisSpotCoverage,
    FilterCoverage,
    IcSpotCoverage,
    SessionConfidenceCoverage,
    compute_session_confidence_coverage,
    resolve_timeslice_confidence_coverage_source,
)

__all__ = [
    "CONFIDENCE_THRESHOLDS",
    "THRESHOLD_STEP",
    "TIMESLICE_CONFIDENCE_COVERAGE_COLS",
    "AxisSpotCoverage",
    "FilterCoverage",
    "IcSpotCoverage",
    "SessionConfidenceCoverage",
    "compute_session_confidence_coverage",
    "resolve_timeslice_confidence_coverage_source",
]
