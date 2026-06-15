"""Spot-level beam-on timeslice coverage versus Gaussian fit filter thresholds."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from . import detect_beam_on_mask, subtract_background_frames
from .g3_timeslice_position import (
    G3PositionTargetColumns,
    G3QualityColumns,
    resolve_g3_position_target_columns,
    resolve_g3_quality_columns,
    resolve_ic_spot_no_column,
    valid_g3_fit_values,
)
from .schema import (
    C_IC1_X_PEAK_AMPLITUDE,
    C_IC1_Y_PEAK_AMPLITUDE,
    C_IC2_X_PEAK_AMPLITUDE,
    C_IC2_Y_PEAK_AMPLITUDE,
    C_LAYER_ID,
    C_TIMESTAMP,
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

_G3_POSITION_TARGET_COLS = (
    "r_ic1_x_position",
    "r_ic1_x_spot_position",
    "ic1_position_x_target",
    "r_ic1_y_position",
    "r_ic1_y_spot_position",
    "ic1_position_y_target",
    "r_ic2_x_position",
    "r_ic2_x_spot_position",
    "ic2_position_x_target",
    "r_ic2_y_position",
    "r_ic2_y_spot_position",
    "ic2_position_y_target",
)

TIMESLICE_GAUSSIAN_FIT_FILTER_COVERAGE_COLS = [
    C_LAYER_ID,
    C_TIMESTAMP,
    "timestamp",
    "time_ms",
    "spot_no",
    "spot_no.1",
    "spot_no.2",
    "rci_in_trigger",
    "r_beamOk",
    *IC_PEAK_AMPLITUDE_COLUMNS,
    *_G3_POSITION_TARGET_COLS,
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
class OrphanSpotPeakSeries:
    """Beam-on peak/error/confidence samples for one orphan spot."""

    layer_id: float
    spot_no: float
    beam_on_index: np.ndarray
    time_ms: np.ndarray | None
    peak_x: np.ndarray
    peak_y: np.ndarray
    error_x: np.ndarray
    error_y: np.ndarray
    confidence_x: np.ndarray
    confidence_y: np.ndarray


@dataclass(frozen=True)
class OrphanSpotPeakSummary:
    """Per-IC peak timeseries for orphan spots (sorted by layer, spot)."""

    ics: dict[str, tuple[OrphanSpotPeakSeries, ...]]


@dataclass(frozen=True)
class IcWeightedPositionRms:
    """Confidence-weighted spot position RMS versus target across a threshold sweep."""

    total_spots: int
    spots_used: np.ndarray
    rms_x_mm: np.ndarray
    rms_y_mm: np.ndarray
    rms_xy_mm: np.ndarray


@dataclass(frozen=True)
class WeightedPositionRmsSweep:
    """Per-IC RMS of confidence×peak weighted position error vs confidence threshold."""

    thresholds: np.ndarray
    ics: dict[str, IcWeightedPositionRms]


@dataclass(frozen=True)
class SessionGaussianFitFilterCoverage:
    confidence: GaussianFitFilterSweep
    peak: GaussianFitFilterSweep | None
    weighted_position_rms: WeightedPositionRmsSweep | None
    error_confidence_issues: ErrorConfidenceIssueSummary
    orphan_spot_errors: OrphanSpotErrorSummary
    orphan_spot_peaks: OrphanSpotPeakSummary


@dataclass(frozen=True)
class _CoverageSource:
    quality: G3QualityColumns
    layer_id: str
    time_col: str | None
    spot_cols: dict[str, str | None]
    peak_cols: dict[str, str | None]
    position_cols: G3PositionTargetColumns | None


_SpotPositionSample = tuple[float, float, float, float]  # position, target, confidence, peak


@dataclass
class _SpotAxisSamples:
    position: np.ndarray
    target: np.ndarray
    confidence: np.ndarray
    peak: np.ndarray
    sorted_conf: np.ndarray | None = None
    suffix_w: np.ndarray | None = None
    suffix_wp: np.ndarray | None = None
    target_val: float | None = None

    @classmethod
    def from_tuples(cls, samples: list[_SpotPositionSample]) -> _SpotAxisSamples:
        if not samples:
            return cls(
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )
        arr = np.asarray(samples, dtype=float)
        return cls(
            position=arr[:, 0],
            target=arr[:, 1],
            confidence=arr[:, 2],
            peak=arr[:, 3],
        )


def _concat_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=float)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)


def _spot_group_starts(layer: np.ndarray, spot: np.ndarray) -> np.ndarray:
    split = (layer[1:] != layer[:-1]) | (spot[1:] != spot[:-1])
    return np.concatenate(([0], np.flatnonzero(split) + 1))


def _unique_spot_keys(layer: np.ndarray, spot: np.ndarray) -> set[tuple[float, float]]:
    if layer.size == 0:
        return set()
    order = np.lexsort((spot, layer))
    layer_s = layer[order]
    spot_s = spot[order]
    starts = _spot_group_starts(layer_s, spot_s)
    return {(float(layer_s[i]), float(spot_s[i])) for i in starts}


def _max_by_spot_arrays(
    layer: np.ndarray,
    spot: np.ndarray,
    value: np.ndarray,
) -> dict[tuple[float, float], float]:
    if layer.size == 0:
        return {}
    order = np.lexsort((spot, layer))
    layer_s = layer[order]
    spot_s = spot[order]
    value_s = value[order]
    starts = _spot_group_starts(layer_s, spot_s)
    max_vals = np.maximum.reduceat(value_s, starts)
    return {
        (float(layer_s[i]), float(spot_s[i])): float(max_vals[j])
        for j, i in enumerate(starts)
    }


def _spot_axis_samples_from_group(
    position: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    peak: np.ndarray,
) -> _SpotAxisSamples | None:
    weight = _axis_peak_weights(confidence, peak)
    valid = (weight > 0) & np.isfinite(position) & np.isfinite(target)
    if not np.any(valid):
        return None
    if not np.isfinite(float(target[valid][0])):
        return None
    return _SpotAxisSamples(
        position=position,
        target=target,
        confidence=confidence,
        peak=peak,
    )


def _with_weighted_error_cache(sample: _SpotAxisSamples) -> _SpotAxisSamples:
    """Attach suffix sums used by the confidence-threshold weighted-error sweep."""
    if sample.suffix_w is not None:
        return sample

    weight = _axis_peak_weights(sample.confidence, sample.peak)
    valid = (weight > 0) & np.isfinite(sample.position) & np.isfinite(sample.target)
    if not np.any(valid):
        return sample

    conf = sample.confidence[valid]
    pos = sample.position[valid]
    tgt = sample.target[valid]
    w = weight[valid]
    target_val = float(tgt[0])
    if not np.isfinite(target_val):
        return sample

    order = np.argsort(conf)
    sorted_conf = conf[order]
    w = w[order]
    wp = w * pos[order]
    return _SpotAxisSamples(
        position=sample.position,
        target=sample.target,
        confidence=sample.confidence,
        peak=sample.peak,
        sorted_conf=sorted_conf,
        suffix_w=np.cumsum(w[::-1])[::-1],
        suffix_wp=np.cumsum(wp[::-1])[::-1],
        target_val=target_val,
    )


@dataclass
class _MetricRowBuffer:
    _layer: list[np.ndarray]
    _spot: list[np.ndarray]
    _value: list[np.ndarray]

    @classmethod
    def empty(cls) -> _MetricRowBuffer:
        return cls([], [], [])

    def append(
        self,
        layer: np.ndarray,
        spot: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        if not np.any(mask):
            return
        self._layer.append(np.asarray(layer[mask], dtype=float))
        self._spot.append(np.asarray(spot[mask], dtype=float))
        self._value.append(np.asarray(value[mask], dtype=float))

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            _concat_chunks(self._layer),
            _concat_chunks(self._spot),
            _concat_chunks(self._value),
        )

    def spot_keys(self) -> set[tuple[float, float]]:
        layer, spot, _ = self.arrays()
        return _unique_spot_keys(layer, spot)

    def max_by_spot(self) -> dict[tuple[float, float], float]:
        layer, spot, value = self.arrays()
        return _max_by_spot_arrays(layer, spot, value)


@dataclass
class _PositionRowBuffer:
    _layer: list[np.ndarray]
    _spot: list[np.ndarray]
    _position: list[np.ndarray]
    _target: list[np.ndarray]
    _confidence: list[np.ndarray]
    _peak: list[np.ndarray]

    @classmethod
    def empty(cls) -> _PositionRowBuffer:
        return cls([], [], [], [], [], [])

    def append(
        self,
        layer: np.ndarray,
        spot: np.ndarray,
        position: np.ndarray,
        target: np.ndarray,
        confidence: np.ndarray,
        peak: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        if not np.any(mask):
            return
        self._layer.append(np.asarray(layer[mask], dtype=float))
        self._spot.append(np.asarray(spot[mask], dtype=float))
        self._position.append(np.asarray(position[mask], dtype=float))
        self._target.append(np.asarray(target[mask], dtype=float))
        self._confidence.append(np.asarray(confidence[mask], dtype=float))
        self._peak.append(np.asarray(peak[mask], dtype=float))

    def arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            _concat_chunks(self._layer),
            _concat_chunks(self._spot),
            _concat_chunks(self._position),
            _concat_chunks(self._target),
            _concat_chunks(self._confidence),
            _concat_chunks(self._peak),
        )


@dataclass
class _IcOrphanTraceBuffer:
    layer: np.ndarray
    spot: np.ndarray
    peak_x: np.ndarray
    peak_y: np.ndarray
    error_x: np.ndarray
    error_y: np.ndarray
    confidence_x: np.ndarray
    confidence_y: np.ndarray
    time: np.ndarray

    @classmethod
    def empty(cls) -> _IcOrphanTraceBuffer:
        empty = np.array([], dtype=float)
        return cls(empty, empty, empty, empty, empty, empty, empty, empty, empty)

    def append_rows(
        self,
        *,
        layer: np.ndarray,
        spot: np.ndarray,
        peak_x: np.ndarray,
        peak_y: np.ndarray,
        error_x: np.ndarray,
        error_y: np.ndarray,
        confidence_x: np.ndarray,
        confidence_y: np.ndarray,
        time: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        if not np.any(mask):
            return
        self.layer = np.concatenate([self.layer, layer[mask]])
        self.spot = np.concatenate([self.spot, spot[mask]])
        self.peak_x = np.concatenate([self.peak_x, peak_x[mask]])
        self.peak_y = np.concatenate([self.peak_y, peak_y[mask]])
        self.error_x = np.concatenate([self.error_x, error_x[mask]])
        self.error_y = np.concatenate([self.error_y, error_y[mask]])
        self.confidence_x = np.concatenate([self.confidence_x, confidence_x[mask]])
        self.confidence_y = np.concatenate([self.confidence_y, confidence_y[mask]])
        self.time = np.concatenate([self.time, time[mask]])


def _finalize_position_samples(
    buffers: dict[str, _PositionRowBuffer],
) -> dict[str, dict[tuple[float, float], _SpotAxisSamples]]:
    out: dict[str, dict[tuple[float, float], _SpotAxisSamples]] = {
        axis: {} for axis in _IC_AXES
    }
    for axis, buf in buffers.items():
        layer, spot, position, target, confidence, peak = buf.arrays()
        if layer.size == 0:
            continue
        order = np.lexsort((spot, layer))
        layer_s = layer[order]
        spot_s = spot[order]
        pos_s = position[order]
        tgt_s = target[order]
        conf_s = confidence[order]
        peak_s = peak[order]
        starts = _spot_group_starts(layer_s, spot_s)
        ends = np.concatenate((starts[1:], [layer_s.size]))
        axis_out: dict[tuple[float, float], _SpotAxisSamples] = {}
        for start, end in zip(starts, ends):
            sample = _spot_axis_samples_from_group(
                pos_s[start:end],
                tgt_s[start:end],
                conf_s[start:end],
                peak_s[start:end],
            )
            if sample is not None:
                axis_out[(float(layer_s[start]), float(spot_s[start]))] = sample
        out[axis] = axis_out
    return out


def _normalize_spot_samples(
    spot_samples: dict[str, dict[tuple[float, float], list[_SpotPositionSample] | _SpotAxisSamples]],
) -> dict[str, dict[tuple[float, float], _SpotAxisSamples]]:
    normalized: dict[str, dict[tuple[float, float], _SpotAxisSamples]] = {}
    for axis, samples in spot_samples.items():
        axis_samples: dict[tuple[float, float], _SpotAxisSamples] = {}
        for key, value in samples.items():
            if isinstance(value, _SpotAxisSamples):
                axis_samples[key] = value
            else:
                axis_samples[key] = _SpotAxisSamples.from_tuples(value)
        normalized[axis] = axis_samples
    return normalized


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
        time_col=resolve_concept_column(columns, C_TIMESTAMP),
        spot_cols=spot_cols,
        peak_cols=peak_cols,
        position_cols=resolve_g3_position_target_columns(columns),
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


def _position_target_values(
    df,
    position_cols: G3PositionTargetColumns,
    axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    meas_col = getattr(position_cols, axis)
    tgt_col = getattr(position_cols, f"{axis}_target")
    if meas_col not in df.columns or tgt_col not in df.columns:
        n = len(df)
        return np.full(n, np.nan), np.full(n, np.nan)
    meas = valid_g3_fit_values(
        pd.to_numeric(df[meas_col], errors="coerce").to_numpy(dtype=float)
    )
    tgt = valid_g3_fit_values(
        pd.to_numeric(df[tgt_col], errors="coerce").to_numpy(dtype=float)
    )
    return meas, tgt


def _confidence_peak_weight(confidence: float, peak: float) -> float:
    if not np.isfinite(confidence) or confidence < 0:
        return 0.0
    if np.isfinite(peak) and peak > 0:
        return float(confidence * peak)
    return float(confidence)


def _axis_peak_weights(confidence: np.ndarray, peak: np.ndarray) -> np.ndarray:
    weight = np.where(
        (confidence >= 0) & np.isfinite(confidence),
        np.where((np.isfinite(peak) & (peak > 0)), confidence * peak, confidence),
        0.0,
    )
    return weight.astype(float, copy=False)


def _axis_errors_at_thresholds(
    confidence: np.ndarray,
    position: np.ndarray,
    target: np.ndarray,
    peak: np.ndarray,
    thresholds: np.ndarray,
    *,
    sample: _SpotAxisSamples | None = None,
) -> np.ndarray:
    """Return confidence×peak weighted position error at each threshold."""
    errors = np.full(len(thresholds), np.nan, dtype=float)
    if (
        sample is not None
        and sample.suffix_w is not None
        and sample.suffix_wp is not None
        and sample.sorted_conf is not None
        and sample.target_val is not None
    ):
        conf = sample.sorted_conf
        suffix_w = sample.suffix_w
        suffix_wp = sample.suffix_wp
        target_val = sample.target_val
    else:
        weight = _axis_peak_weights(confidence, peak)
        valid = (weight > 0) & np.isfinite(position) & np.isfinite(target)
        if not np.any(valid):
            return errors

        conf = confidence[valid]
        pos = position[valid]
        tgt = target[valid]
        w = weight[valid]
        target_val = float(tgt[0])
        if not np.isfinite(target_val):
            return errors

        order = np.argsort(conf)
        conf = conf[order]
        w = w[order]
        wp = w * pos[order]
        suffix_w = np.cumsum(w[::-1])[::-1]
        suffix_wp = np.cumsum(wp[::-1])[::-1]

    starts = np.searchsorted(conf, thresholds, side="left")
    n = len(conf)
    ok = starts < n
    safe = np.minimum(starts, n - 1)
    total_w = suffix_w[safe]
    total_wp = suffix_wp[safe]
    weighted_pos = np.where(ok & (total_w > 0), total_wp / total_w, np.nan)
    return weighted_pos - target_val


def _weighted_axis_position_error(
    samples: list[_SpotPositionSample],
    threshold: float,
) -> float | None:
    """Return confidence×peak weighted position minus target for one axis."""
    if not samples:
        return None
    axis = _SpotAxisSamples.from_tuples(samples)
    err = _axis_errors_at_thresholds(
        axis.confidence,
        axis.position,
        axis.target,
        axis.peak,
        np.asarray([threshold], dtype=float),
    )[0]
    if not np.isfinite(err):
        return None
    return float(err)


def _build_weighted_position_rms_sweep(
    spot_samples: dict[
        str, dict[tuple[float, float], list[_SpotPositionSample] | _SpotAxisSamples]
    ],
    thresholds: np.ndarray,
) -> WeightedPositionRmsSweep | None:
    normalized = _normalize_spot_samples(spot_samples)
    ics: dict[str, IcWeightedPositionRms] = {}
    for ic in _ICS:
        x_axis, y_axis = _IC_AXIS_PAIRS[ic]
        x_samples = normalized[x_axis]
        y_samples = normalized[y_axis]
        spots = set(x_samples) | set(y_samples)
        if not spots:
            continue

        total = len(spots)
        err_x_rows: list[np.ndarray] = []
        err_y_rows: list[np.ndarray] = []
        for spot in spots:
            xs = x_samples.get(spot)
            ys = y_samples.get(spot)
            if xs is None or ys is None:
                continue
            xs = _with_weighted_error_cache(xs)
            ys = _with_weighted_error_cache(ys)
            ex = _axis_errors_at_thresholds(
                xs.confidence,
                xs.position,
                xs.target,
                xs.peak,
                thresholds,
                sample=xs,
            )
            ey = _axis_errors_at_thresholds(
                ys.confidence,
                ys.position,
                ys.target,
                ys.peak,
                thresholds,
                sample=ys,
            )
            if not np.any(np.isfinite(ex) & np.isfinite(ey)):
                continue
            err_x_rows.append(ex)
            err_y_rows.append(ey)

        if not err_x_rows:
            continue

        err_x_mat = np.vstack(err_x_rows)
        err_y_mat = np.vstack(err_y_rows)
        valid = np.isfinite(err_x_mat) & np.isfinite(err_y_mat)
        spots_used = valid.sum(axis=0).astype(int)
        sum_x2 = np.sum(np.where(valid, err_x_mat**2, 0.0), axis=0)
        sum_y2 = np.sum(np.where(valid, err_y_mat**2, 0.0), axis=0)
        sum_xy2 = np.sum(
            np.where(valid, err_x_mat**2 + err_y_mat**2, 0.0),
            axis=0,
        )
        used = spots_used > 0
        denom = spots_used.astype(float)
        rms_x = np.full_like(denom, np.nan)
        rms_y = np.full_like(denom, np.nan)
        rms_xy = np.full_like(denom, np.nan)
        np.divide(sum_x2, denom, out=rms_x, where=used)
        np.divide(sum_y2, denom, out=rms_y, where=used)
        np.divide(sum_xy2, denom, out=rms_xy, where=used)
        np.sqrt(rms_x, out=rms_x, where=used)
        np.sqrt(rms_y, out=rms_y, where=used)
        np.sqrt(rms_xy, out=rms_xy, where=used)

        ics[ic] = IcWeightedPositionRms(
            total_spots=total,
            spots_used=spots_used,
            rms_x_mm=rms_x.astype(float),
            rms_y_mm=rms_y.astype(float),
            rms_xy_mm=rms_xy.astype(float),
        )

    if not ics:
        return None
    return WeightedPositionRmsSweep(thresholds=thresholds, ics=ics)


def _numeric_column(df, col: str | None) -> np.ndarray:
    if col is None or col not in df.columns:
        return np.full(len(df), np.nan, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _process_beam_on_frame(
    df,
    source: _CoverageSource,
    *,
    conf_all_rows: dict[str, _MetricRowBuffer],
    conf_metric_rows: dict[str, _MetricRowBuffer],
    peak_all_rows: dict[str, _MetricRowBuffer],
    peak_metric_rows: dict[str, _MetricRowBuffer],
    position_rows: dict[str, _PositionRowBuffer],
    issue_counts: dict[str, np.ndarray],
    has_peak_cols: bool,
) -> None:
    """Single pass over one beam-on frame for all coverage accumulators."""
    layer = _numeric_column(df, source.layer_id)
    position_cols = source.position_cols
    quality = source.quality
    peak_by_axis: dict[str, np.ndarray] = {}
    spot_by_ic: dict[str, np.ndarray] = {}
    for ic in _ICS:
        spot_by_ic[ic] = _numeric_column(df, source.spot_cols.get(ic))

    for axis in _IC_AXES:
        ic = _AXIS_IC[axis]
        spot = spot_by_ic[ic]
        spot_ok = np.isfinite(layer) & np.isfinite(spot)
        if not np.any(spot_ok):
            continue

        conf_all_rows[axis].append(layer, spot, spot, spot_ok)
        gate = _row_passes_base_gates(df, quality, axis)

        confidence = _numeric_column(df, getattr(quality, f"{axis}_confidence"))
        conf_qual = spot_ok & gate & np.isfinite(confidence) & (confidence >= 0)
        conf_metric_rows[axis].append(layer, spot, confidence, conf_qual)

        if has_peak_cols:
            peak_all_rows[axis].append(layer, spot, spot, spot_ok)
            peak = _numeric_column(df, source.peak_cols.get(axis))
            peak_by_axis[axis] = peak
            peak_qual = spot_ok & gate & np.isfinite(peak) & (peak > 0)
            peak_metric_rows[axis].append(layer, spot, peak, peak_qual)

        codes = _numeric_column(df, getattr(quality, f"{axis}_error_code"))
        code_ok = np.isfinite(codes) & (codes >= 0) & (codes <= SPOT_ERROR_CODE_MAX)
        issue = code_ok & _confidence_invalid(confidence)
        if np.any(issue):
            code_idx = np.round(codes[issue]).astype(int)
            np.add.at(issue_counts[axis], code_idx, 1)

        if position_cols is not None:
            position, target = _position_target_values(df, position_cols, axis)
            peak = peak_by_axis.get(axis)
            if peak is None:
                peak = _numeric_column(df, source.peak_cols.get(axis))
            pos_qual = (
                spot_ok
                & gate
                & np.isfinite(position)
                & np.isfinite(target)
                & np.isfinite(confidence)
                & (confidence >= 0)
            )
            position_rows[axis].append(
                layer, spot, position, target, confidence, peak, pos_qual
            )


def _orphan_ic_spot_set(
    all_spots: dict[str, set[tuple[float, float]]],
    max_metric: dict[str, dict[tuple[float, float], float]],
    ic: str,
) -> set[tuple[float, float]]:
    """Spots with beam-on data but no good combined X/Y fit at confidence threshold 0."""
    _spots, combined = _combined_ic_spots_and_metrics(all_spots, max_metric, ic)
    return {spot for spot in _spots if combined.get(spot, -np.inf) < 0.0}


def _orphan_mask(
    layer: np.ndarray,
    spot: np.ndarray,
    orphans: set[tuple[float, float]],
) -> np.ndarray:
    mask = np.zeros(len(layer), dtype=bool)
    for layer_id, spot_no in orphans:
        mask |= (layer == layer_id) & (spot == spot_no)
    return mask


def _orphan_row_counts_from_trace(
    trace: _IcOrphanTraceBuffer,
    orphans: set[tuple[float, float]],
) -> np.ndarray:
    counts = np.zeros(len(SPOT_ERROR_CODES), dtype=int)
    if not orphans or trace.layer.size == 0:
        return counts

    mask = _orphan_mask(trace.layer, trace.spot, orphans)
    if not np.any(mask):
        return counts

    for codes in (trace.error_x[mask], trace.error_y[mask]):
        code_ok = np.isfinite(codes) & (codes >= 0) & (codes <= SPOT_ERROR_CODE_MAX)
        if not np.any(code_ok):
            continue
        code_idx = np.round(codes[code_ok]).astype(int)
        np.add.at(counts, code_idx, 1)
    return counts


def _collect_orphan_ic_traces(
    frames,
    source: _CoverageSource,
    orphan_spots: dict[str, set[tuple[float, float]]],
) -> dict[str, _IcOrphanTraceBuffer]:
    traces = {ic: _IcOrphanTraceBuffer.empty() for ic in _ICS}
    if not any(orphan_spots.values()):
        return traces
    if not any(col is not None for col in source.peak_cols.values()):
        return traces

    has_time = source.time_col is not None
    for df in frames:
        beam_on = detect_beam_on_mask(df)
        if beam_on is None:
            continue
        frame = df.loc[beam_on]
        layer = pd.to_numeric(frame[source.layer_id], errors="coerce").to_numpy(dtype=float)
        time_vals = (
            _column_values(frame, source.time_col)
            if has_time and source.time_col in frame.columns
            else np.full(len(frame), np.nan, dtype=float)
        )

        for ic in _ICS:
            orphans = orphan_spots.get(ic)
            if not orphans:
                continue
            x_axis, y_axis = _IC_AXIS_PAIRS[ic]
            spot_col = source.spot_cols.get(ic)
            if spot_col is None or spot_col not in frame.columns:
                continue

            spot = pd.to_numeric(frame[spot_col], errors="coerce").to_numpy(dtype=float)
            mask = _orphan_mask(layer, spot, orphans)
            if not np.any(mask):
                continue

            traces[ic].append_rows(
                layer=layer,
                spot=spot,
                peak_x=_peak_values(frame, source.peak_cols, x_axis),
                peak_y=_peak_values(frame, source.peak_cols, y_axis),
                error_x=_error_code_values(frame, source.quality, x_axis),
                error_y=_error_code_values(frame, source.quality, y_axis),
                confidence_x=_confidence_values(frame, source.quality, x_axis),
                confidence_y=_confidence_values(frame, source.quality, y_axis),
                time=time_vals,
                mask=mask,
            )
    return traces


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


def _column_values(df, col_name: str) -> np.ndarray:
    data = df[col_name]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    return pd.to_numeric(data, errors="coerce").to_numpy(dtype=float)


def _spot_time_ms(values: np.ndarray, *, time_col: str | None) -> np.ndarray | None:
    if time_col is None:
        return None
    times = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(times)):
        return None
    t0 = float(np.nanmin(times))
    return times - t0


def _build_orphan_spot_peak_series(
    ic_traces: dict[str, _IcOrphanTraceBuffer],
    source: _CoverageSource,
    orphan_spots: dict[str, set[tuple[float, float]]],
) -> OrphanSpotPeakSummary:
    """Build per-row peak and fit-quality samples for each orphan spot."""
    if not any(col is not None for col in source.peak_cols.values()):
        return OrphanSpotPeakSummary(ics={ic: () for ic in _ICS})

    has_time = source.time_col is not None
    ics: dict[str, list[OrphanSpotPeakSeries]] = {ic: [] for ic in _ICS}

    for ic in _ICS:
        orphans = orphan_spots.get(ic)
        if not orphans:
            continue
        trace = ic_traces[ic]
        if trace.layer.size == 0:
            continue

        for layer_id, spot_no in sorted(orphans):
            mask = (trace.layer == layer_id) & (trace.spot == spot_no)
            if not np.any(mask):
                continue

            peak_x = trace.peak_x[mask]
            peak_y = trace.peak_y[mask]
            err_x = trace.error_x[mask]
            err_y = trace.error_y[mask]
            conf_x = trace.confidence_x[mask]
            conf_y = trace.confidence_y[mask]
            n = len(peak_x)

            time_ms = None
            if has_time:
                time_rows = trace.time[mask]
                if len(time_rows) == n:
                    time_ms = _spot_time_ms(time_rows, time_col=source.time_col)

            ics[ic].append(
                OrphanSpotPeakSeries(
                    layer_id=float(layer_id),
                    spot_no=float(spot_no),
                    beam_on_index=np.arange(n, dtype=int),
                    time_ms=time_ms,
                    peak_x=peak_x,
                    peak_y=peak_y,
                    error_x=err_x,
                    error_y=err_y,
                    confidence_x=conf_x,
                    confidence_y=conf_y,
                )
            )

    return OrphanSpotPeakSummary(
        ics={ic: tuple(series_list) for ic, series_list in ics.items()}
    )


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

    conf_all_rows = {axis: _MetricRowBuffer.empty() for axis in _IC_AXES}
    conf_metric_rows = {axis: _MetricRowBuffer.empty() for axis in _IC_AXES}
    peak_all_rows = {axis: _MetricRowBuffer.empty() for axis in _IC_AXES}
    peak_metric_rows = {axis: _MetricRowBuffer.empty() for axis in _IC_AXES}
    position_rows = {axis: _PositionRowBuffer.empty() for axis in _IC_AXES}
    issue_counts: dict[str, np.ndarray] = {
        axis: np.zeros(len(SPOT_ERROR_CODES), dtype=int) for axis in _IC_AXES
    }

    has_peak_cols = any(col is not None for col in source.peak_cols.values())

    for df in frames:
        beam_on = detect_beam_on_mask(df)
        if beam_on is None:
            continue
        _process_beam_on_frame(
            df.loc[beam_on],
            source,
            conf_all_rows=conf_all_rows,
            conf_metric_rows=conf_metric_rows,
            peak_all_rows=peak_all_rows,
            peak_metric_rows=peak_metric_rows,
            position_rows=position_rows,
            issue_counts=issue_counts,
            has_peak_cols=has_peak_cols,
        )

    conf_spots = {axis: buf.spot_keys() for axis, buf in conf_all_rows.items()}
    conf_max = {axis: buf.max_by_spot() for axis, buf in conf_metric_rows.items()}
    peak_spots = {axis: buf.spot_keys() for axis, buf in peak_all_rows.items()}
    peak_max = {axis: buf.max_by_spot() for axis, buf in peak_metric_rows.items()}

    confidence = _build_filter_coverage(
        "confidence", CONFIDENCE_THRESHOLDS, conf_spots, conf_max
    )
    if confidence is None:
        return None

    error_confidence_issues = _build_error_confidence_issues(issue_counts)

    weighted_position_rms = None
    if source.position_cols is not None:
        position_samples = _finalize_position_samples(position_rows)
        weighted_position_rms = _build_weighted_position_rms_sweep(
            position_samples, CONFIDENCE_THRESHOLDS
        )

    orphan_spot_sets = {
        ic: _orphan_ic_spot_set(conf_spots, conf_max, ic) for ic in _ICS
    }
    if any(orphan_spot_sets.values()):
        ic_traces = _collect_orphan_ic_traces(frames, source, orphan_spot_sets)
    else:
        ic_traces = {ic: _IcOrphanTraceBuffer.empty() for ic in _ICS}
    orphan_row_counts = {
        ic: _orphan_row_counts_from_trace(ic_traces[ic], orphan_spot_sets[ic])
        for ic in _ICS
    }
    orphan_spot_errors = _build_orphan_spot_errors(orphan_spot_sets, orphan_row_counts)
    orphan_spot_peaks = _build_orphan_spot_peak_series(
        ic_traces, source, orphan_spot_sets
    )

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
        weighted_position_rms=weighted_position_rms,
        error_confidence_issues=error_confidence_issues,
        orphan_spot_errors=orphan_spot_errors,
        orphan_spot_peaks=orphan_spot_peaks,
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
