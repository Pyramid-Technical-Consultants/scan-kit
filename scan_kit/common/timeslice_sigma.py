"""Beam-on timeslice IC sigma (spot size) loading for G2 and G3 sessions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from . import detect_beam_on_mask, subtract_background_frames
from .g3_timeslice_position import G3QualityColumns, resolve_g3_quality_columns
from .schema import C_LAYER_ID, resolve_column_name
from .session_source import (
    load_session_timeslice_device_units,
    resolve_session_source,
)

_log = logging.getLogger(__name__)

_SIGMA_SPECS = (
    ("ic1_x", ("r_ic1_x_sigma",)),
    ("ic1_y", ("r_ic1_y_sigma",)),
    ("ic2_x", ("r_ic2_x_sigma",)),
    ("ic2_y", ("r_ic2_y_sigma",)),
)

_G2_IC2_X_SIGMA_OK_ALIASES = ("r_x_sigma_ok.1", "r_x_sigma_ok")
_G2_IC2_Y_SIGMA_OK_ALIASES = ("r_y_sigma_ok.1", "r_y_sigma_ok")

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

_MAX_SIGMA_MM = 20.0

TIMESLICE_SIGMA_COLS = [
    C_LAYER_ID,
    "rci_in_trigger",
    "r_beamOk",
    *(name for _, aliases in _SIGMA_SPECS for name in aliases),
    *_G2_IC2_X_SIGMA_OK_ALIASES,
    *_G2_IC2_Y_SIGMA_OK_ALIASES,
    *_G3_QUALITY_COLS,
]


@dataclass(frozen=True)
class SessionIcSigmas:
    ic1_x: np.ndarray
    ic1_y: np.ndarray
    ic2_x: np.ndarray
    ic2_y: np.ndarray


@dataclass(frozen=True)
class _DirectSigmaSource:
    mode: Literal["direct"]
    columns: dict[str, str]
    ic2_x_ok: str | None = None
    ic2_y_ok: str | None = None


@dataclass(frozen=True)
class _G3SigmaSource:
    mode: Literal["g3"]
    columns: dict[str, str]
    quality: G3QualityColumns


TimesliceSigmaSource = _DirectSigmaSource | _G3SigmaSource


def _resolve_first_column(columns, aliases: tuple[str, ...]) -> str | None:
    for name in aliases:
        col = resolve_column_name(columns, name)
        if col is not None:
            return col
    return None


def _resolve_sigma_columns(columns) -> dict[str, str] | None:
    resolved: dict[str, str] = {}
    for label, aliases in _SIGMA_SPECS:
        col = _resolve_first_column(columns, aliases)
        if col is None:
            return None
        resolved[label] = col
    return resolved


def _is_g3_sigma_session(columns) -> bool:
    quality = resolve_g3_quality_columns(columns)
    return any(
        col is not None
        for col in (
            quality.ic1_x_fit_ok,
            quality.ic1_x_confidence,
            quality.ic1_x_error_code,
        )
    )


def resolve_timeslice_sigma_source(columns) -> TimesliceSigmaSource | None:
    """Resolve sigma columns from one timeslice frame header (no session I/O)."""
    sigma_cols = _resolve_sigma_columns(columns)
    if sigma_cols is None:
        return None
    if _is_g3_sigma_session(columns):
        return _G3SigmaSource("g3", sigma_cols, resolve_g3_quality_columns(columns))
    return _DirectSigmaSource(
        "direct",
        sigma_cols,
        ic2_x_ok=_resolve_first_column(columns, _G2_IC2_X_SIGMA_OK_ALIASES),
        ic2_y_ok=_resolve_first_column(columns, _G2_IC2_Y_SIGMA_OK_ALIASES),
    )


def _sanitize_sigma(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(float, copy=True)
    out[~np.isfinite(out)] = np.nan
    out[out <= 0] = np.nan
    out[out > _MAX_SIGMA_MM] = np.nan
    return out


def _quality_mask(df, quality: G3QualityColumns, axis: str) -> np.ndarray | None:
    fit_ok = getattr(quality, f"{axis}_fit_ok")
    confidence = getattr(quality, f"{axis}_confidence")
    error_code = getattr(quality, f"{axis}_error_code")
    if fit_ok is None and confidence is None and error_code is None:
        return None

    n = len(df)
    mask = np.ones(n, dtype=bool)
    if fit_ok is not None and fit_ok in df.columns:
        mask &= pd.to_numeric(df[fit_ok], errors="coerce").fillna(0).to_numpy() != 0
    if confidence is not None and confidence in df.columns:
        from .g3_timeslice_position import G3_FIT_CONFIDENCE_MIN

        conf = pd.to_numeric(df[confidence], errors="coerce").to_numpy(dtype=float)
        mask &= np.isfinite(conf) & (conf >= G3_FIT_CONFIDENCE_MIN)
    if error_code is not None and error_code in df.columns:
        codes = pd.to_numeric(df[error_code], errors="coerce").fillna(1).to_numpy()
        mask &= codes == 0
    return mask


def _apply_mask(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return values
    out = values.copy()
    out[~mask] = np.nan
    return out


def _apply_ic2_sigma_ok_gate(
    values: np.ndarray,
    df,
    ok_col: str | None,
) -> np.ndarray:
    if ok_col is None:
        return values
    ok = pd.to_numeric(df[ok_col], errors="coerce").values
    gated = values.copy()
    gated[ok != 1] = np.nan
    return gated


def frame_timeslice_sigma_arrays(
    df,
    source: TimesliceSigmaSource,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if source.mode == "g3":
        arrays = {}
        axis_keys = ("ic1_x", "ic1_y", "ic2_x", "ic2_y")
        for key in axis_keys:
            raw = _sanitize_sigma(df[source.columns[key]].values)
            raw = _apply_mask(raw, _quality_mask(df, source.quality, key))
            arrays[key] = raw
    else:
        ic2_x = _sanitize_sigma(df[source.columns["ic2_x"]].values)
        ic2_y = _sanitize_sigma(df[source.columns["ic2_y"]].values)
        ic2_x = _apply_ic2_sigma_ok_gate(ic2_x, df, source.ic2_x_ok)
        ic2_y = _apply_ic2_sigma_ok_gate(ic2_y, df, source.ic2_y_ok)
        arrays = {
            "ic1_x": _sanitize_sigma(df[source.columns["ic1_x"]].values),
            "ic1_y": _sanitize_sigma(df[source.columns["ic1_y"]].values),
            "ic2_x": ic2_x,
            "ic2_y": ic2_y,
        }

    if not any(np.isfinite(v).any() for v in arrays.values()):
        return None
    return arrays["ic1_x"], arrays["ic1_y"], arrays["ic2_x"], arrays["ic2_y"]


def load_session_beam_on_sigmas(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
) -> SessionIcSigmas | None:
    """Load all hardware-gated beam-on sigma samples for one session."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    frames = load_session_timeslice_device_units(src, usecols=TIMESLICE_SIGMA_COLS)
    if not frames:
        return None
    if bg_subtract:
        subtract_background_frames(frames)

    source = resolve_timeslice_sigma_source(frames[0].columns)
    if source is None:
        return None

    parts: dict[str, list[np.ndarray]] = {
        key: [] for key in ("ic1_x", "ic1_y", "ic2_x", "ic2_y")
    }

    for df in frames:
        beam_on = detect_beam_on_mask(df)
        if beam_on is None:
            continue

        frame_sigmas = frame_timeslice_sigma_arrays(df, source)
        if frame_sigmas is None:
            continue

        ic1_x, ic1_y, ic2_x, ic2_y = frame_sigmas
        ic1_x, ic1_y, ic2_x, ic2_y = (
            ic1_x[beam_on],
            ic1_y[beam_on],
            ic2_x[beam_on],
            ic2_y[beam_on],
        )
        if not np.isfinite(ic1_x).any() and not np.isfinite(ic2_x).any():
            continue

        parts["ic1_x"].append(ic1_x)
        parts["ic1_y"].append(ic1_y)
        parts["ic2_x"].append(ic2_x)
        parts["ic2_y"].append(ic2_y)

    if not parts["ic1_x"]:
        return None

    return SessionIcSigmas(
        ic1_x=np.concatenate(parts["ic1_x"]),
        ic1_y=np.concatenate(parts["ic1_y"]),
        ic2_x=np.concatenate(parts["ic2_x"]),
        ic2_y=np.concatenate(parts["ic2_y"]),
    )
