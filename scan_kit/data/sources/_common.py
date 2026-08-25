"""Shared probe helpers for data-source modules."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ...common import C_X_POSITION, C_Y_POSITION, detect_beam_on_mask, resolve_concept_column
from ...common.schema import POSITION_KEY_G2, POSITION_KEY_G3
from ...common.session_source import (
    load_session_timeslice_device_units,
    read_session_csv_columns,
    resolve_session_source,
)
from ..context import SessionContext
from ..spot import spot_ic_position_columns_available


def probe_spot_positions_with_plan(ctx: SessionContext) -> bool:
    src = resolve_session_source(ctx.session_id, ctx.base_dir)
    if src is None:
        return False
    input_cols = read_session_csv_columns(src, "input_map.csv")
    spot_cols = read_session_csv_columns(src, "spot_data.csv")
    if not input_cols or not spot_cols:
        return False
    if resolve_concept_column(input_cols, C_X_POSITION) is None:
        return False
    if resolve_concept_column(input_cols, C_Y_POSITION) is None:
        return False
    return any(
        spot_ic_position_columns_available(spot_cols, position_key)
        for position_key in (POSITION_KEY_G3, POSITION_KEY_G2)
    )


def probe_timeslice_frame(
    ctx: SessionContext,
    *,
    usecols: list[str],
    resolve_source: Callable,
    frame_arrays: Callable,
) -> bool:
    src = resolve_session_source(ctx.session_id, ctx.base_dir)
    if src is None:
        return False
    frames = load_session_timeslice_device_units(src, usecols=usecols, max_frames=1)
    if not frames:
        return False
    source = resolve_source(frames[0].columns)
    if source is None:
        return False
    beam_on = detect_beam_on_mask(frames[0])
    if beam_on is None or not np.any(beam_on):
        return False
    return frame_arrays(frames[0], source) is not None


def probe_timeslice_session_arrays(
    ctx: SessionContext,
    *,
    usecols: list[str],
    resolve_session_fn: Callable,
    frame_arrays_fn: Callable,
) -> bool:
    src = resolve_session_source(ctx.session_id, ctx.base_dir)
    if src is None:
        return False
    frames = load_session_timeslice_device_units(src, usecols=usecols, max_frames=1)
    if not frames:
        return False
    source = resolve_session_fn(src, frames)
    if source is None:
        return False
    beam_on = detect_beam_on_mask(frames[0])
    if beam_on is None or not np.any(beam_on):
        return False
    return frame_arrays_fn(frames[0], source) is not None
