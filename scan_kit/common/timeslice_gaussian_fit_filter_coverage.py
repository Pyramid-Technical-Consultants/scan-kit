"""Spot-level beam-on timeslice coverage versus Gaussian fit filter thresholds."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from . import detect_beam_on_mask, subtract_background_frames
from .g3_timeslice_position import (
    G3QualityColumns,
    resolve_g3_quality_columns,
    resolve_ic_spot_no_column,
)
from .schema import (
    C_IC1_X_PEAK_AMPLITUDE,
    C_IC1_Y_PEAK_AMPLITUDE,
    C_IC2_X_PEAK_AMPLITUDE,
    C_IC2_Y_PEAK_AMPLITUDE,
    C_LAYER_ID,
    IC_PEAK_AMPLITUDE_COLUMNS,
    resolve_concept_column,
    resolve_column_name,
)
from .session_source import (
    load_session_timeslice_device_units,
    resolve_session_source,
)

_log = logging.getLogger(__name__)

FilterKind = Literal["confidence", "peak"]

_IC_AXES = ("ic1_x", "ic1_y", "ic2_x", "ic2_y")
_ICS = ("ic1", "ic2")
_IC_AXIS_PAIRS = {
    "ic1": ("ic1_x", "ic1_y"),
    "ic2": ("ic2_x", "ic2_y"),
}
_AXIS_IC = {
    "ic1_x": "ic1",
    "ic1_y": "ic1",
    "ic2_x": "ic2",
    "ic2_y": "ic2",
}

_PEAK_CONCEPTS = {
    "ic1_x": C_IC1_X_PEAK_AMPLITUDE,
    "ic1_y": C_IC1_Y_PEAK_AMPLITUDE,
    "ic2_x": C_IC2_X_PEAK_AMPLITUDE,
    "ic2_y": C_IC2_Y_PEAK_AMPLITUDE,
}

_G3_QUALITY_COLS = (
    "ic1_x_fit_ok",
    "ic1_y_fit_ok",
    "ic2_x_fit_ok",
    "ic2_y_fit_ok",
    "r_ic1_x_confidence",
    "r_ic1_y_confidence",
    "r_ic2_x_confidence",
    "r_ic2_y_confidence",
    "r_ic1_x_spot_error_code",
    "r_ic1_y_spot_error_code",
    "r_ic2_x_spot_error_code",
    "r_ic2_y_spot_error_code",
)

TIMESLICE_GAUSSIAN_FIT_FILTER_COVERAGE_COLS = [
    C_LAYER_ID,
    "spot_no",
    "spot_no.1",
    "spot_no.2",
    "rci_in_trigger",
    "r_beamOk",
    *IC_PEAK_AMPLITUDE_COLUMNS,
    *_G3_QUALITY_COLS,
]

CONFIDENCE_THRESHOLD_STEP = 0.25
CONFIDENCE_THRESHOLDS = np.arange(
    0.0, 100.0 + CONFIDENCE_THRESHOLD_STEP / 2, CONFIDENCE_THRESHOLD_STEP
)
PEAK_THRESHOLD_POINTS = len(CONFIDENCE_THRESHOLDS)
PEAK_THRESHOLD_PERCENTILE = 99.95
SPOT_ERROR_CODE_MAX = 5
SPOT_ERROR_CODES = np.arange(0, SPOT_ERROR_CODE_MAX + 1, dtype=int)
# Error-confidence issue bars omit routine codes 0/1 (none / peak below threshold).
PLOTTED_ERROR_CONFIDENCE_ISSUE_CODES = np.array([2, 3, 4, 5], dtype=int)
# Backward-compatible alias for axis tick positions.
ERROR_CODE_THRESHOLDS = SPOT_ERROR_CODES.astype(float)

# G3 Gaussian-fit spot_error_code values (ion_chamber / strip fit).
SPOT_ERROR_CODE_NONE = 0
SPOT_ERROR_CODE_PEAK_BELOW_THRESHOLD = 1
SPOT_ERROR_CODE_PEAK_EDGE = 2
SPOT_ERROR_CODE_NOT_ENOUGH_PEAK_DATA = 3
SPOT_ERROR_CODE_MATRIX_NON_INVERTIBLE = 4
SPOT_ERROR_CODE_SOLUTION_INVALID = 5

SPOT_ERROR_CODE_NAMES: dict[int, str] = {
    SPOT_ERROR_CODE_NONE: "No error",
    SPOT_ERROR_CODE_PEAK_BELOW_THRESHOLD: "Peak below threshold",
    SPOT_ERROR_CODE_PEAK_EDGE: "Peak at edge",
    SPOT_ERROR_CODE_NOT_ENOUGH_PEAK_DATA: "Not enough peak data",
    SPOT_ERROR_CODE_MATRIX_NON_INVERTIBLE: "Matrix non-invertible",
    SPOT_ERROR_CODE_SOLUTION_INVALID: "Solution invalid",
}

SPOT_ERROR_CODE_AXIS_LABELS: dict[int, str] = {
    SPOT_ERROR_CODE_NONE: "0 None",
    SPOT_ERROR_CODE_PEAK_BELOW_THRESHOLD: "1 Peak<th",
    SPOT_ERROR_CODE_PEAK_EDGE: "2 Edge",
    SPOT_ERROR_CODE_NOT_ENOUGH_PEAK_DATA: "3 Few peaks",
    SPOT_ERROR_CODE_MATRIX_NON_INVERTIBLE: "4 Singular",
    SPOT_ERROR_CODE_SOLUTION_INVALID: "5 Bad sol.",
}


def spot_error_code_name(code: float | int) -> str:
    """Return the documented Gaussian-fit error label for integer *code*."""
    key = int(round(float(code)))
    return SPOT_ERROR_CODE_NAMES.get(key, f"Code {key}")


@dataclass(frozen=True)
class IcSpotCoverage:
    """Coverage curve for one IC (combined X/Y) across a swept Gaussian fit filter threshold."""

    total_spots: int
    coverage_pct: np.ndarray
    full_coverage_breakpoint: float | None


@dataclass(frozen=True)
class GaussianFitFilterSweep:
    kind: FilterKind
    thresholds: np.ndarray
    ics: dict[str, IcSpotCoverage]

    @property
    def axes(self) -> dict[str, IcSpotCoverage]:
        """Backward-compatible alias for :attr:`ics`."""
        return self.ics


@dataclass(frozen=True)
class IcErrorConfidenceIssueCounts:
    """Beam-on row counts with invalid confidence, grouped by spot error code."""

    counts_by_code: np.ndarray


@dataclass(frozen=True)
class ErrorConfidenceIssueSummary:
    """Per-IC counts of beam-on rows with a spot error code and invalid confidence."""

    codes: np.ndarray
    ics: dict[str, IcErrorConfidenceIssueCounts]


@dataclass(frozen=True)
class IcOrphanSpotErrorCounts:
    """Beam-on row counts of spot error codes on orphan spots (combined X/Y)."""

    orphan_spots: int
    counts_by_code: np.ndarray


@dataclass(frozen=True)
class OrphanSpotErrorSummary:
    """Per-IC orphan spots that lack good fit data even at zero confidence threshold."""

    codes: np.ndarray
    ics: dict[str, IcOrphanSpotErrorCounts]


@dataclass(frozen=True)
class SessionGaussianFitFilterCoverage:
    confidence: GaussianFitFilterSweep
    peak: GaussianFitFilterSweep | None
    error_confidence_issues: ErrorConfidenceIssueSummary
    orphan_spot_errors: OrphanSpotErrorSummary


@dataclass(frozen=True)
class _CoverageSource:
    quality: G3QualityColumns
    layer_id: str
    spot_cols: dict[str, str | None]
    peak_cols: dict[str, str | None]


def _has_confidence_columns(quality: G3QualityColumns) -> bool:
    return any(
        col is not None
        for col in (
            quality.ic1_x_confidence,
            quality.ic1_y_confidence,
            quality.ic2_x_confidence,
            quality.ic2_y_confidence,
        )
    )


def resolve_timeslice_gaussian_fit_filter_coverage_source(columns) -> _CoverageSource | None:
    """Resolve columns needed for Gaussian fit filter spot coverage."""
    quality = resolve_g3_quality_columns(columns)
    if not _has_confidence_columns(quality):
        return None

    layer_id = resolve_column_name(columns, C_LAYER_ID)
    if layer_id is None:
        return None

    spot_cols = {
        ic: resolve_ic_spot_no_column(columns, ic) for ic in ("ic1", "ic2")
    }
    if spot_cols["ic1"] is None and spot_cols["ic2"] is None:
        return None

    peak_cols = {
        axis: resolve_concept_column(columns, concept)
        for axis, concept in _PEAK_CONCEPTS.items()
    }

    return _CoverageSource(
        quality=quality,
        layer_id=layer_id,
        spot_cols=spot_cols,
        peak_cols=peak_cols,
    )


def _row_passes_fit_ok(df, quality: G3QualityColumns, axis: str) -> np.ndarray:
    """Apply fit_ok when present; otherwise leave rows eligible (matches G3 quality mask)."""
    fit_ok = getattr(quality, f"{axis}_fit_ok")
    n = len(df)
    if fit_ok is None or fit_ok not in df.columns:
        return np.ones(n, dtype=bool)
    return pd.to_numeric(df[fit_ok], errors="coerce").fillna(0).to_numpy() != 0


def _row_passes_base_gates(
    df,
    quality: G3QualityColumns,
    axis: str,
) -> np.ndarray:
    mask = _row_passes_fit_ok(df, quality, axis)
    error_code = getattr(quality, f"{axis}_error_code")
    if error_code is not None and error_code in df.columns:
        codes = pd.to_numeric(df[error_code], errors="coerce").fillna(1).to_numpy()
        mask &= codes == 0
    return mask


def _confidence_values(df, quality: G3QualityColumns, axis: str) -> np.ndarray:
    col = getattr(quality, f"{axis}_confidence")
    if col is None or col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _confidence_invalid(values: np.ndarray) -> np.ndarray:
    """True when confidence is missing or uses the G3 invalid sentinel (< 0)."""
    return ~np.isfinite(values) | (values < 0)


def _error_code_values(df, quality: G3QualityColumns, axis: str) -> np.ndarray:
    col = getattr(quality, f"{axis}_error_code")
    if col is None or col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _peak_values(df, peak_cols: dict[str, str | None], axis: str) -> np.ndarray:
    col = peak_cols.get(axis)
    if col is None or col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _accumulate_axis_spots(
    accum: dict[str, dict[tuple[float, float], float]],
    all_spots: dict[str, set[tuple[float, float]]],
    df,
    source: _CoverageSource,
    *,
    kind: FilterKind,
) -> None:
    layer = pd.to_numeric(df[source.layer_id], errors="coerce").to_numpy(dtype=float)

    for axis in _IC_AXES:
        ic = _AXIS_IC[axis]
        spot_col = source.spot_cols.get(ic)
        if spot_col is None or spot_col not in df.columns:
            continue

        spot = pd.to_numeric(df[spot_col], errors="coerce").to_numpy(dtype=float)
        spot_ok = np.isfinite(layer) & np.isfinite(spot)
        if not np.any(spot_ok):
            continue

        all_spots[axis].update(zip(layer[spot_ok], spot[spot_ok]))

        gate = _row_passes_base_gates(df, source.quality, axis)
        if kind == "confidence":
            metric = _confidence_values(df, source.quality, axis)
            qualifying = spot_ok & gate & np.isfinite(metric) & (metric >= 0)
        else:
            metric = _peak_values(df, source.peak_cols, axis)
            qualifying = spot_ok & gate & np.isfinite(metric) & (metric > 0)

        if not np.any(qualifying):
            continue

        axis_max = accum[axis]
        for key, value in zip(
            zip(layer[qualifying], spot[qualifying]),
            metric[qualifying],
        ):
            prev = axis_max.get(key)
            if prev is None or value > prev:
                axis_max[key] = float(value)


def _accumulate_invalid_confidence_by_error_code(
    counts: dict[str, np.ndarray],
    df,
    source: _CoverageSource,
) -> None:
    """Count beam-on rows with a spot error code and invalid confidence."""
    for axis in _IC_AXES:
        codes = _error_code_values(df, source.quality, axis)
        confidence = _confidence_values(df, source.quality, axis)
        code_ok = np.isfinite(codes) & (codes >= 0) & (codes <= SPOT_ERROR_CODE_MAX)
        issue = code_ok & _confidence_invalid(confidence)
        if not np.any(issue):
            continue
        for code in codes[issue]:
            counts[axis][int(round(code))] += 1


def _orphan_ic_spot_set(
    all_spots: dict[str, set[tuple[float, float]]],
    max_metric: dict[str, dict[tuple[float, float], float]],
    ic: str,
) -> set[tuple[float, float]]:
    """Spots with beam-on data but no good combined X/Y fit at confidence threshold 0."""
    _spots, combined = _combined_ic_spots_and_metrics(all_spots, max_metric, ic)
    return {spot for spot in _spots if combined.get(spot, -np.inf) < 0.0}


def _accumulate_orphan_row_error_codes(
    counts: dict[str, np.ndarray],
    df,
    source: _CoverageSource,
    orphan_spots: dict[str, set[tuple[float, float]]],
) -> None:
    """Count every spot error code on beam-on rows belonging to orphan spots."""
    layer = pd.to_numeric(df[source.layer_id], errors="coerce").to_numpy(dtype=float)

    for axis in _IC_AXES:
        ic = _AXIS_IC[axis]
        orphans = orphan_spots.get(ic)
        if not orphans:
            continue

        spot_col = source.spot_cols.get(ic)
        if spot_col is None or spot_col not in df.columns:
            continue

        spot = pd.to_numeric(df[spot_col], errors="coerce").to_numpy(dtype=float)
        spot_ok = np.isfinite(layer) & np.isfinite(spot)
        if not np.any(spot_ok):
            continue

        codes = _error_code_values(df, source.quality, axis)
        code_ok = (
            spot_ok
            & np.isfinite(codes)
            & (codes >= 0)
            & (codes <= SPOT_ERROR_CODE_MAX)
        )
        if not np.any(code_ok):
            continue

        ic_counts = counts[ic]
        for key, code in zip(
            zip(layer[code_ok], spot[code_ok]),
            codes[code_ok],
        ):
            if key not in orphans:
                continue
            ic_counts[int(round(code))] += 1


def _coverage_curve(
    all_spots: set[tuple[float, float]],
    max_metric: dict[tuple[float, float], float],
    thresholds: np.ndarray,
) -> IcSpotCoverage:
    total = len(all_spots)
    if total == 0:
        return IcSpotCoverage(0, np.zeros_like(thresholds), None)

    max_vals = np.array(
        [max_metric.get(spot, -np.inf) for spot in all_spots],
        dtype=float,
    )
    covered = max_vals[:, None] >= thresholds[None, :]
    pct = 100.0 * covered.mean(axis=0)
    finite_max = max_vals[np.isfinite(max_vals)]
    breakpoint = float(np.min(finite_max)) if finite_max.size else None
    return IcSpotCoverage(total, pct.astype(float), breakpoint)


def _peak_thresholds(max_metric: dict[str, dict[tuple[float, float], float]]) -> np.ndarray | None:
    values = [
        v
        for axis_vals in max_metric.values()
        for v in axis_vals.values()
        if np.isfinite(v) and v > 0
    ]
    if not values:
        return None
    hi = float(np.percentile(values, PEAK_THRESHOLD_PERCENTILE))
    if hi <= 0:
        hi = 1.0
    return np.linspace(0.0, hi, PEAK_THRESHOLD_POINTS)


def _combined_ic_spots_and_metrics(
    all_spots: dict[str, set[tuple[float, float]]],
    max_metric: dict[str, dict[tuple[float, float], float]],
    ic: str,
) -> tuple[set[tuple[float, float]], dict[tuple[float, float], float]]:
    """Merge X/Y per spot: covered only when both axes have qualifying data."""
    x_axis, y_axis = _IC_AXIS_PAIRS[ic]
    spots = all_spots[x_axis] | all_spots[y_axis]
    combined: dict[tuple[float, float], float] = {}
    for spot in spots:
        mx = max_metric[x_axis].get(spot, -np.inf)
        my = max_metric[y_axis].get(spot, -np.inf)
        combined[spot] = float(min(mx, my))
    return spots, combined


def _combine_ic_error_confidence_counts(
    axis_counts: dict[str, np.ndarray],
    ic: str,
) -> np.ndarray:
    """Sum X/Y beam-on row counts for one IC."""
    x_axis, y_axis = _IC_AXIS_PAIRS[ic]
    return axis_counts[x_axis] + axis_counts[y_axis]


def _build_filter_coverage(
    kind: FilterKind,
    thresholds: np.ndarray,
    all_spots: dict[str, set[tuple[float, float]]],
    max_metric: dict[str, dict[tuple[float, float], float]],
) -> GaussianFitFilterSweep | None:
    ics: dict[str, IcSpotCoverage] = {}
    for ic in _ICS:
        spots, combined = _combined_ic_spots_and_metrics(all_spots, max_metric, ic)
        if not spots:
            continue
        ics[ic] = _coverage_curve(spots, combined, thresholds)

    if not ics:
        return None
    return GaussianFitFilterSweep(kind=kind, thresholds=thresholds, ics=ics)


def _build_error_confidence_issues(
    axis_counts: dict[str, np.ndarray],
) -> ErrorConfidenceIssueSummary:
    ics = {
        ic: IcErrorConfidenceIssueCounts(
            counts_by_code=_combine_ic_error_confidence_counts(axis_counts, ic).copy()
        )
        for ic in _ICS
    }
    return ErrorConfidenceIssueSummary(codes=SPOT_ERROR_CODES.copy(), ics=ics)


def _build_orphan_spot_errors(
    orphan_spots: dict[str, set[tuple[float, float]]],
    counts: dict[str, np.ndarray],
) -> OrphanSpotErrorSummary:
    ics = {
        ic: IcOrphanSpotErrorCounts(
            orphan_spots=len(orphan_spots.get(ic, ())),
            counts_by_code=counts[ic].copy(),
        )
        for ic in _ICS
    }
    return OrphanSpotErrorSummary(codes=SPOT_ERROR_CODES.copy(), ics=ics)


def compute_session_gaussian_fit_filter_coverage(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
) -> SessionGaussianFitFilterCoverage | None:
    """Compute per-IC (combined X/Y) spot coverage versus Gaussian fit filter thresholds."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_GAUSSIAN_FIT_FILTER_COVERAGE_COLS
    )
    if not frames:
        return None
    if bg_subtract:
        subtract_background_frames(frames)

    source = resolve_timeslice_gaussian_fit_filter_coverage_source(frames[0].columns)
    if source is None:
        return None

    conf_spots: dict[str, set[tuple[float, float]]] = {axis: set() for axis in _IC_AXES}
    conf_max: dict[str, dict[tuple[float, float], float]] = {
        axis: {} for axis in _IC_AXES
    }
    peak_spots: dict[str, set[tuple[float, float]]] = {axis: set() for axis in _IC_AXES}
    peak_max: dict[str, dict[tuple[float, float], float]] = {
        axis: {} for axis in _IC_AXES
    }
    issue_counts: dict[str, np.ndarray] = {
        axis: np.zeros(len(SPOT_ERROR_CODES), dtype=int) for axis in _IC_AXES
    }

    has_peak_cols = any(col is not None for col in source.peak_cols.values())

    for df in frames:
        beam_on = detect_beam_on_mask(df)
        if beam_on is None:
            continue
        frame = df.loc[beam_on]
        _accumulate_axis_spots(
            conf_max, conf_spots, frame, source, kind="confidence"
        )
        _accumulate_invalid_confidence_by_error_code(issue_counts, frame, source)
        if has_peak_cols:
            _accumulate_axis_spots(
                peak_max, peak_spots, frame, source, kind="peak"
            )

    confidence = _build_filter_coverage(
        "confidence", CONFIDENCE_THRESHOLDS, conf_spots, conf_max
    )
    if confidence is None:
        return None

    error_confidence_issues = _build_error_confidence_issues(issue_counts)

    orphan_spot_sets = {
        ic: _orphan_ic_spot_set(conf_spots, conf_max, ic) for ic in _ICS
    }
    orphan_row_counts = {
        ic: np.zeros(len(SPOT_ERROR_CODES), dtype=int) for ic in _ICS
    }
    for df in frames:
        beam_on = detect_beam_on_mask(df)
        if beam_on is None:
            continue
        _accumulate_orphan_row_error_codes(
            orphan_row_counts, df.loc[beam_on], source, orphan_spot_sets
        )
    orphan_spot_errors = _build_orphan_spot_errors(orphan_spot_sets, orphan_row_counts)

    peak = None
    if has_peak_cols:
        peak_thresholds = _peak_thresholds(peak_max)
        if peak_thresholds is not None:
            peak = _build_filter_coverage(
                "peak", peak_thresholds, peak_spots, peak_max
            )

    return SessionGaussianFitFilterCoverage(
        confidence=confidence,
        peak=peak,
        error_confidence_issues=error_confidence_issues,
        orphan_spot_errors=orphan_spot_errors,
    )


# Backward-compatible aliases.
AxisSpotCoverage = IcSpotCoverage
FilterCoverage = GaussianFitFilterSweep
SessionGaussianFilterCoverage = SessionGaussianFitFilterCoverage
SessionConfidenceCoverage = GaussianFitFilterSweep
TIMESLICE_FILTER_COVERAGE_COLS = TIMESLICE_GAUSSIAN_FIT_FILTER_COVERAGE_COLS
TIMESLICE_CONFIDENCE_COVERAGE_COLS = TIMESLICE_GAUSSIAN_FIT_FILTER_COVERAGE_COLS
THRESHOLD_STEP = CONFIDENCE_THRESHOLD_STEP


def compute_session_gaussian_filter_coverage(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
) -> SessionGaussianFitFilterCoverage | None:
    return compute_session_gaussian_fit_filter_coverage(
        session_id, base_dir, bg_subtract=bg_subtract
    )


def resolve_timeslice_filter_coverage_source(columns) -> _CoverageSource | None:
    return resolve_timeslice_gaussian_fit_filter_coverage_source(columns)


def compute_session_confidence_coverage(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
    thresholds: np.ndarray | None = None,
) -> GaussianFitFilterSweep | None:
    """Compute per-IC spot coverage versus confidence threshold only."""
    _ = thresholds
    coverage = compute_session_gaussian_fit_filter_coverage(
        session_id, base_dir, bg_subtract=bg_subtract
    )
    if coverage is None:
        return None
    return coverage.confidence


def resolve_timeslice_confidence_coverage_source(columns) -> _CoverageSource | None:
    return resolve_timeslice_gaussian_fit_filter_coverage_source(columns)
