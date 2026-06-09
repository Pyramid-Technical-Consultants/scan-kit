"""Timeslice IC position error loading (G2 direct errors, G3 measured minus target)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import detect_beam_on_mask, subtract_background_frames
from .g3_timeslice_position import (
    G3PositionTargetColumns,
    g3_position_error_frame_arrays,
    resolve_g3_position_target_columns,
)
from .schema import C_LAYER_ID, resolve_column_name
from .session_source import load_session_timeslice_device_units, resolve_session_source

_G2_DIRECT_ERROR = (
    ("ic1_x_err", "position_err_X"),
    ("ic1_y_err", "position_err_Y"),
    ("ic2_x_err", "position_err_X2"),
    ("ic2_y_err", "position_err_Y2"),
)

TIMESLICE_POSITION_ERROR_COLS = [
    C_LAYER_ID,
    "rci_in_trigger",
    "r_beamOk",
    *(name for _, name in _G2_DIRECT_ERROR),
    "r_ic1_x_position",
    "ic1_position_x_target",
    "r_ic1_y_position",
    "ic1_position_y_target",
    "r_ic2_x_position",
    "ic2_position_x_target",
    "r_ic2_y_position",
    "ic2_position_y_target",
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


@dataclass(frozen=True)
class _G3PositionTargetSource:
    mode: Literal["g3_position_target"]
    columns: G3PositionTargetColumns


TimesliceErrorSource = _DirectErrorSource | _G3PositionTargetSource


def _resolve_named_columns(columns, pairs: tuple[tuple[str, str], ...]) -> dict[str, str] | None:
    resolved: dict[str, str] = {}
    for label, name in pairs:
        col = resolve_column_name(columns, name)
        if col is None:
            return None
        resolved[label] = col
    return resolved


def resolve_timeslice_error_source(columns) -> TimesliceErrorSource | None:
    direct = _resolve_named_columns(columns, _G2_DIRECT_ERROR)
    if direct is not None:
        return _DirectErrorSource("direct", direct)

    g3_cols = resolve_g3_position_target_columns(columns)
    if g3_cols is not None:
        return _G3PositionTargetSource("g3_position_target", g3_cols)

    return None


def _sanitize_error(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(float, copy=True)
    out[~np.isfinite(out)] = np.nan
    out[np.abs(out) > 128] = np.nan
    return out


def frame_timeslice_error_arrays(
    df,
    source: TimesliceErrorSource,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if source.mode == "direct":
        return (
            _sanitize_error(df[source.columns["ic1_x_err"]].values),
            _sanitize_error(df[source.columns["ic1_y_err"]].values),
            _sanitize_error(df[source.columns["ic2_x_err"]].values),
            _sanitize_error(df[source.columns["ic2_y_err"]].values),
        )

    return g3_position_error_frame_arrays(df, source.columns)


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

    error_source = resolve_timeslice_error_source(frames[0].columns)
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
