"""Beam-on timeslice IC fit-confidence loading for G3 (partial G2)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from . import detect_beam_on_mask, subtract_background_frames
from .g3_timeslice_position import G3QualityColumns, resolve_g3_quality_columns
from .schema import (
    C_IC1_CURRENT,
    C_IC1_X_PEAK_AMPLITUDE,
    C_IC1_Y_PEAK_AMPLITUDE,
    C_IC2_CURRENT,
    C_IC2_X_PEAK_AMPLITUDE,
    C_IC2_Y_PEAK_AMPLITUDE,
    C_LAYER_ID,
    IC_PEAK_AMPLITUDE_COLUMNS,
    resolve_concept_column,
)
from .session_source import (
    load_session_timeslice_device_units,
    resolve_session_source,
)

_log = logging.getLogger(__name__)

_CONFIDENCE_AXES = ("ic1_x", "ic1_y", "ic2_x", "ic2_y")

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

TIMESLICE_CONFIDENCE_COLS = [
    C_LAYER_ID,
    "rci_in_trigger",
    "r_beamOk",
    C_IC1_CURRENT,
    C_IC2_CURRENT,
    *IC_PEAK_AMPLITUDE_COLUMNS,
    *_G3_QUALITY_COLS,
]


@dataclass(frozen=True)
class SessionConfidenceCorrelations:
    """Aligned beam-on samples for confidence vs peak-current / primary-channel."""

    ic1_x_conf: np.ndarray
    ic1_y_conf: np.ndarray
    ic2_x_conf: np.ndarray
    ic2_y_conf: np.ndarray
    ic1_x_peak: np.ndarray
    ic1_y_peak: np.ndarray
    ic2_x_peak: np.ndarray
    ic2_y_peak: np.ndarray
    ic1_primary: np.ndarray
    ic2_primary: np.ndarray
    has_peak: bool


@dataclass(frozen=True)
class _ConfidenceSource:
    quality: G3QualityColumns
    peak_cols: dict[str, str | None]
    ic1_primary: str | None
    ic2_primary: str | None


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


def _sanitize_confidence(values: np.ndarray) -> np.ndarray:
    out = values.astype(float, copy=True)
    out[~np.isfinite(out)] = np.nan
    out[out < 0] = np.nan
    return out


def _sanitize_positive(values: np.ndarray) -> np.ndarray:
    out = values.astype(float, copy=True)
    out[~np.isfinite(out)] = np.nan
    out[out <= 0] = np.nan
    return out


def resolve_timeslice_confidence_source(columns) -> _ConfidenceSource | None:
    """Resolve confidence and correlated columns from one timeslice header."""
    quality = resolve_g3_quality_columns(columns)
    if not _has_confidence_columns(quality):
        return None

    peak_cols = {
        axis: resolve_concept_column(columns, concept)
        for axis, concept in _PEAK_CONCEPTS.items()
    }
    return _ConfidenceSource(
        quality=quality,
        peak_cols=peak_cols,
        ic1_primary=resolve_concept_column(columns, C_IC1_CURRENT),
        ic2_primary=resolve_concept_column(columns, C_IC2_CURRENT),
    )


def _confidence_column(quality: G3QualityColumns, axis: str) -> str | None:
    return getattr(quality, f"{axis}_confidence")


def _frame_arrays(
    df,
    source: _ConfidenceSource,
) -> dict[str, np.ndarray] | None:
    conf: dict[str, np.ndarray] = {}
    for axis in _CONFIDENCE_AXES:
        col = _confidence_column(source.quality, axis)
        if col is None or col not in df.columns:
            conf[axis] = np.full(len(df), np.nan)
        else:
            conf[axis] = _sanitize_confidence(df[col].to_numpy(dtype=float))

    if not any(np.isfinite(v).any() for v in conf.values()):
        return None

    peak: dict[str, np.ndarray] = {}
    for axis in _CONFIDENCE_AXES:
        col = source.peak_cols[axis]
        if col is None or col not in df.columns:
            peak[axis] = np.full(len(df), np.nan)
        else:
            peak[axis] = _sanitize_positive(df[col].to_numpy(dtype=float))

    ic1_primary = (
        _sanitize_positive(df[source.ic1_primary].to_numpy(dtype=float))
        if source.ic1_primary is not None and source.ic1_primary in df.columns
        else np.full(len(df), np.nan)
    )
    ic2_primary = (
        _sanitize_positive(df[source.ic2_primary].to_numpy(dtype=float))
        if source.ic2_primary is not None and source.ic2_primary in df.columns
        else np.full(len(df), np.nan)
    )

    return {
        **{f"{axis}_conf": conf[axis] for axis in _CONFIDENCE_AXES},
        **{f"{axis}_peak": peak[axis] for axis in _CONFIDENCE_AXES},
        "ic1_primary": ic1_primary,
        "ic2_primary": ic2_primary,
    }


def load_session_beam_on_confidence_correlations(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
) -> SessionConfidenceCorrelations | None:
    """Load beam-on confidence samples and correlated IC metrics for one session."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    frames = load_session_timeslice_device_units(src, usecols=TIMESLICE_CONFIDENCE_COLS)
    if not frames:
        return None
    if bg_subtract:
        subtract_background_frames(frames)

    source = resolve_timeslice_confidence_source(frames[0].columns)
    if source is None:
        return None

    parts: dict[str, list[np.ndarray]] = {
        "ic1_x_conf": [],
        "ic1_y_conf": [],
        "ic2_x_conf": [],
        "ic2_y_conf": [],
        "ic1_x_peak": [],
        "ic1_y_peak": [],
        "ic2_x_peak": [],
        "ic2_y_peak": [],
        "ic1_primary": [],
        "ic2_primary": [],
    }

    for df in frames:
        beam_on = detect_beam_on_mask(df)
        if beam_on is None:
            continue

        arrays = _frame_arrays(df, source)
        if arrays is None:
            continue

        sliced = {key: values[beam_on] for key, values in arrays.items()}
        if not any(np.isfinite(sliced[f"{axis}_conf"]).any() for axis in _CONFIDENCE_AXES):
            continue

        for key, values in sliced.items():
            parts[key].append(values)

    if not parts["ic1_x_conf"]:
        return None

    merged = {key: np.concatenate(chunks) for key, chunks in parts.items()}
    has_peak = any(np.isfinite(merged[f"{axis}_peak"]).any() for axis in _CONFIDENCE_AXES)

    return SessionConfidenceCorrelations(
        ic1_x_conf=merged["ic1_x_conf"],
        ic1_y_conf=merged["ic1_y_conf"],
        ic2_x_conf=merged["ic2_x_conf"],
        ic2_y_conf=merged["ic2_y_conf"],
        ic1_x_peak=merged["ic1_x_peak"],
        ic1_y_peak=merged["ic1_y_peak"],
        ic2_x_peak=merged["ic2_x_peak"],
        ic2_y_peak=merged["ic2_y_peak"],
        ic1_primary=merged["ic1_primary"],
        ic2_primary=merged["ic2_primary"],
        has_peak=has_peak,
    )
