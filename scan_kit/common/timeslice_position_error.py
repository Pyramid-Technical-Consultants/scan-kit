"""Timeslice IC position error loading (G2 direct errors, G3 isocentric plan)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from . import detect_beam_on_mask, subtract_background_frames
from .g3_timeslice_position import (
    G3IsoErrorContext,
    G3PositionTargetColumns,
    build_g3_iso_error_context,
    g3_iso_position_error_frame_arrays,
    resolve_g3_position_target_columns,
)
from .schema import C_LAYER_ID, C_SPOT_NO, resolve_column_name
from .session_source import (
    SessionSource,
    load_session_csv,
    load_session_timeslice_device_units,
    resolve_session_source,
)

_log = logging.getLogger(__name__)

_G2_IC1_X_ERROR_ALIASES = ("x_err_filtered", "position_err_X", "pos_err_x")
_G2_IC1_Y_ERROR_ALIASES = ("y_err_filtered", "position_err_Y", "pos_err_y")
_G2_IC2_X_ERROR_ALIASES = ("position_err_X2", "pos_err_x2")
_G2_IC2_Y_ERROR_ALIASES = ("position_err_Y2", "pos_err_y2")
_G2_IC2_X_OK_ALIASES = ("r_x_position_ok.1", "r_x_position_ok")
_G2_IC2_Y_OK_ALIASES = ("r_y_position_ok.1", "r_y_position_ok")

_G2_ERROR_SPECS = (
    ("ic1_x_err", _G2_IC1_X_ERROR_ALIASES),
    ("ic1_y_err", _G2_IC1_Y_ERROR_ALIASES),
    ("ic2_x_err", _G2_IC2_X_ERROR_ALIASES),
    ("ic2_y_err", _G2_IC2_Y_ERROR_ALIASES),
)

_G2_ERROR_SENTINELS = (1000.0, -1000.0, 10000.0, -10000.0)
_G2_MAX_ABS_ERROR_MM = 10.0

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

TIMESLICE_POSITION_ERROR_COLS = [
    C_LAYER_ID,
    C_SPOT_NO,
    "spot_no.1",
    "spot_no.2",
    "rci_in_trigger",
    "r_beamOk",
    *(
        name
        for aliases in (
            _G2_IC1_X_ERROR_ALIASES,
            _G2_IC1_Y_ERROR_ALIASES,
            _G2_IC2_X_ERROR_ALIASES,
            _G2_IC2_Y_ERROR_ALIASES,
        )
        for name in aliases
    ),
    *_G2_IC2_X_OK_ALIASES,
    *_G2_IC2_Y_OK_ALIASES,
    "x_err_filtered",
    "y_err_filtered",
    "r_ic1_x_position",
    "ic1_position_x_target",
    "r_ic1_y_position",
    "ic1_position_y_target",
    "r_ic2_x_position",
    "ic2_position_x_target",
    "r_ic2_y_position",
    "ic2_position_y_target",
    *_G3_QUALITY_COLS,
]


@dataclass(frozen=True)
class SessionPositionErrors:
    ic1_x: np.ndarray
    ic1_y: np.ndarray
    ic2_x: np.ndarray
    ic2_y: np.ndarray


@dataclass(frozen=True)
class _DirectErrorSource:
    mode: Literal["direct"]
    columns: dict[str, str]
    ic2_x_ok: str | None = None
    ic2_y_ok: str | None = None


@dataclass(frozen=True)
class _G3PositionTargetSource:
    mode: Literal["g3_position_target"]
    columns: G3PositionTargetColumns


@dataclass(frozen=True)
class _G3IsoPositionSource:
    mode: Literal["g3_iso"]
    context: G3IsoErrorContext


TimesliceErrorSource = _DirectErrorSource | _G3PositionTargetSource | _G3IsoPositionSource


def _resolve_first_column(columns, aliases: tuple[str, ...]) -> str | None:
    for name in aliases:
        col = resolve_column_name(columns, name)
        if col is not None:
            return col
    return None


def _resolve_g2_direct_source(columns) -> _DirectErrorSource | None:
    resolved: dict[str, str] = {}
    for label, aliases in _G2_ERROR_SPECS:
        col = _resolve_first_column(columns, aliases)
        if col is None:
            return None
        resolved[label] = col
    return _DirectErrorSource(
        "direct",
        resolved,
        ic2_x_ok=_resolve_first_column(columns, _G2_IC2_X_OK_ALIASES),
        ic2_y_ok=_resolve_first_column(columns, _G2_IC2_Y_OK_ALIASES),
    )


def resolve_timeslice_error_source(columns) -> TimesliceErrorSource | None:
    """Resolve error columns from one timeslice frame header (no session I/O)."""
    direct = _resolve_g2_direct_source(columns)
    if direct is not None:
        return direct

    g3_cols = resolve_g3_position_target_columns(columns)
    if g3_cols is not None:
        return _G3PositionTargetSource("g3_position_target", g3_cols)

    return None


def resolve_session_timeslice_error_source(
    src: SessionSource,
    frames: list,
) -> TimesliceErrorSource | None:
    """Resolve the error source for one session (G2 direct or G3 isocentric only)."""
    if not frames:
        return None

    base = resolve_timeslice_error_source(frames[0].columns)
    if base is None:
        return None
    if base.mode != "g3_position_target":
        return base

    input_map = load_session_csv(src, "input_map.csv")
    if input_map is None:
        _log.debug("Session %s: missing input_map for G3 iso errors", src.session_id)
        return None

    spot_data = load_session_csv(src, "spot_data.csv")
    iso_ctx = build_g3_iso_error_context(
        input_map, frames, base.columns, spot_data=spot_data
    )
    if iso_ctx is None:
        _log.debug(
            "Session %s: could not build G3 isocentric error context",
            src.session_id,
        )
        return None
    return _G3IsoPositionSource("g3_iso", iso_ctx)


def _sanitize_error(arr: np.ndarray, *, max_abs: float = 128.0) -> np.ndarray:
    out = arr.astype(float, copy=True)
    out[~np.isfinite(out)] = np.nan
    out[np.abs(out) > max_abs] = np.nan
    return out


def _sanitize_g2_error(arr: np.ndarray) -> np.ndarray:
    out = _sanitize_error(arr, max_abs=_G2_MAX_ABS_ERROR_MM)
    for sentinel in _G2_ERROR_SENTINELS:
        out[out == sentinel] = np.nan
    return out


def _apply_ic2_position_ok_gate(
    errors: np.ndarray,
    df,
    ok_col: str | None,
) -> np.ndarray:
    if ok_col is None:
        return errors
    ok = pd.to_numeric(df[ok_col], errors="coerce").values
    gated = errors.copy()
    gated[ok != 1] = np.nan
    return gated


def frame_timeslice_error_arrays(
    df,
    source: TimesliceErrorSource,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if source.mode == "direct":
        ic2_x = _sanitize_g2_error(df[source.columns["ic2_x_err"]].values)
        ic2_y = _sanitize_g2_error(df[source.columns["ic2_y_err"]].values)
        ic2_x = _apply_ic2_position_ok_gate(ic2_x, df, source.ic2_x_ok)
        ic2_y = _apply_ic2_position_ok_gate(ic2_y, df, source.ic2_y_ok)
        return (
            _sanitize_g2_error(df[source.columns["ic1_x_err"]].values),
            _sanitize_g2_error(df[source.columns["ic1_y_err"]].values),
            ic2_x,
            ic2_y,
        )

    if source.mode == "g3_iso":
        arrays = g3_iso_position_error_frame_arrays(df, source.context)
    else:
        _log.warning("Unexpected G3 device-frame error source; skipping frame")
        return None

    if arrays is None:
        return None
    return tuple(_sanitize_error(arr) for arr in arrays)


def _beam_on_slices(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    beam_on: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ic1_x, ic1_y, ic2_x, ic2_y = arrays
    return ic1_x[beam_on], ic1_y[beam_on], ic2_x[beam_on], ic2_y[beam_on]


def load_session_beam_on_position_errors(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
) -> SessionPositionErrors | None:
    """Load all hardware-gated beam-on position error samples for one session."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS
    )
    if not frames:
        return None
    if bg_subtract:
        subtract_background_frames(frames)

    error_source = resolve_session_timeslice_error_source(src, frames)
    if error_source is None:
        return None

    ic1_x_parts: list[np.ndarray] = []
    ic1_y_parts: list[np.ndarray] = []
    ic2_x_parts: list[np.ndarray] = []
    ic2_y_parts: list[np.ndarray] = []

    for df in frames:
        beam_on = detect_beam_on_mask(df)
        if beam_on is None:
            continue

        frame_errors = frame_timeslice_error_arrays(df, error_source)
        if frame_errors is None:
            continue

        ic1_x, ic1_y, ic2_x, ic2_y = _beam_on_slices(frame_errors, beam_on)
        if not np.isfinite(ic1_x).any() and not np.isfinite(ic2_x).any():
            continue

        ic1_x_parts.append(ic1_x)
        ic1_y_parts.append(ic1_y)
        ic2_x_parts.append(ic2_x)
        ic2_y_parts.append(ic2_y)

    if not ic1_x_parts:
        return None

    return SessionPositionErrors(
        ic1_x=np.concatenate(ic1_x_parts),
        ic1_y=np.concatenate(ic1_y_parts),
        ic2_x=np.concatenate(ic2_x_parts),
        ic2_y=np.concatenate(ic2_y_parts),
    )
