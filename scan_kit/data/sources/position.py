"""Spot and timeslice IC position (with plan when available)."""

from __future__ import annotations

import numpy as np

from ...common import C_X_POSITION, C_Y_POSITION, detect_beam_on_mask, process_position_data, try_load_position_data
from ...common.session_source import load_session_timeslice_device_units, resolve_session_source
from ...common.timeslice_position_error import TIMESLICE_POSITION_ERROR_COLS
from ..context import LoadOptions, SessionContext
from ..reference_frame import (
    spot_positions_raw,
    timeslice_position_loader,
    timeslice_position_table_hooks,
)
from ..registry import DataSourceSpec, register
from ..types import (
    GRANULARITY_SPOT,
    GRANULARITY_TIMESLICE_SAMPLE,
    REFERENCE_CHAMBER,
    REFERENCE_ISO,
)
from ._common import probe_spot_positions_with_plan

SOURCE_POSITION = "position"


def _probe_timeslice(ctx: SessionContext, reference_frame: str) -> bool:
    src = resolve_session_source(ctx.session_id, ctx.base_dir)
    if src is None:
        return False
    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS, max_frames=1,
    )
    if not frames:
        return False
    prepare, extract, _keys = timeslice_position_table_hooks(reference_frame)
    source = prepare(src, frames)
    if source is None:
        return False
    beam_on = detect_beam_on_mask(frames[0])
    if beam_on is None or not np.any(beam_on):
        return False
    return extract(frames[0], source) is not None


def probe_position(ctx: SessionContext, opts: LoadOptions) -> bool:
    if opts.granularity == GRANULARITY_SPOT:
        return probe_spot_positions_with_plan(ctx)
    if opts.granularity == GRANULARITY_TIMESLICE_SAMPLE:
        return _probe_timeslice(ctx, opts.reference_frame)
    return False


def _load_spot(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    def _loader(sid, position_key, bdir):
        return process_position_data(
            sid,
            position_key,
            extra_input_columns=[C_X_POSITION, C_Y_POSITION],
            base_dir=bdir,
        )

    data = try_load_position_data(
        ctx.session_id,
        ctx.base_dir,
        _loader,
        raw=spot_positions_raw(opts.reference_frame),
    )
    if data is None:
        return None
    if C_X_POSITION not in data or C_Y_POSITION not in data:
        return None
    return {
        "session_id": ctx.session_id,
        "plan_x": np.asarray(data[C_X_POSITION], dtype=float),
        "plan_y": np.asarray(data[C_Y_POSITION], dtype=float),
        "ic1_x": np.asarray(data["ic1_x"], dtype=float),
        "ic1_y": np.asarray(data["ic1_y"], dtype=float),
        "ic2_x": np.asarray(data["ic2_x"], dtype=float),
        "ic2_y": np.asarray(data["ic2_y"], dtype=float),
    }


def _load_timeslice(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    bg_subtract = opts.resolved_bg_subtract(ctx)
    loader = timeslice_position_loader(opts.reference_frame)
    positions = loader(ctx.session_id, ctx.base_dir, bg_subtract=bg_subtract)
    if positions is None:
        return None
    out: dict = {
        "session_id": ctx.session_id,
        "ic1_x": np.asarray(positions.ic1_x, dtype=float),
        "ic1_y": np.asarray(positions.ic1_y, dtype=float),
        "ic2_x": np.asarray(positions.ic2_x, dtype=float),
        "ic2_y": np.asarray(positions.ic2_y, dtype=float),
        "beam_on": positions.beam_on,
    }
    if positions.plan_x is not None:
        out["plan_x"] = np.asarray(positions.plan_x, dtype=float)
    if positions.plan_y is not None:
        out["plan_y"] = np.asarray(positions.plan_y, dtype=float)
    return out


def load_position(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    if opts.granularity == GRANULARITY_SPOT:
        return _load_spot(ctx, opts)
    if opts.granularity == GRANULARITY_TIMESLICE_SAMPLE:
        return _load_timeslice(ctx, opts)
    return None


SPEC = register(
    DataSourceSpec(
        id=SOURCE_POSITION,
        label="Position",
        granularities=frozenset({GRANULARITY_SPOT, GRANULARITY_TIMESLICE_SAMPLE}),
        reference_frames=frozenset({REFERENCE_ISO, REFERENCE_CHAMBER}),
        supports_bg_subtract=True,
        supports_beam_filter=True,
        probe=probe_position,
        load=load_position,
    )
)
