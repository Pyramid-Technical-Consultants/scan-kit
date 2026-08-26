"""IC2 minus IC1 position difference (spot and timeslice samples)."""

from __future__ import annotations

import numpy as np

from ...common import C_ENERGY, detect_beam_on_mask, process_position_data, try_load_position_data
from ...common.session_source import (
    load_session_timeslice_device_units,
    read_session_csv_columns,
    resolve_session_source,
)
from ...common.timeslice_position_error import TIMESLICE_POSITION_ERROR_COLS
from ...common.timeslice_table import load_energy_tagged_table
from ..context import LoadOptions, SessionContext
from ..reference_frame import (
    spot_positions_raw,
    timeslice_position_loader,
    timeslice_position_table_hooks,
)
from ..registry import DataSourceSpec, register
from ..spot import spot_has_ic_positions
from ..types import (
    DATA_SOURCE_SPOT_ISO,
    DATA_SOURCE_SPOT_CHAMBER,
    DATA_SOURCE_TIMESLICE_ISO,
    DATA_SOURCE_TIMESLICE_CHAMBER,
    GRANULARITY_SPOT,
    GRANULARITY_TIMESLICE_SAMPLE,
)

SOURCE_IC12_POS_DIFF = "ic12_pos_diff"


def _probe_spot(ctx: SessionContext) -> bool:
    src = resolve_session_source(ctx.session_id, ctx.base_dir)
    if src is None:
        return False
    spot_cols = read_session_csv_columns(src, "spot_data.csv")
    if not spot_cols:
        return False
    return spot_has_ic_positions(spot_cols)


def _probe_timeslice_frame(ctx: SessionContext, reference_frame: str) -> bool:
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


def probe_ic12(ctx: SessionContext, opts: LoadOptions) -> bool:
    if opts.granularity == GRANULARITY_SPOT:
        return _probe_spot(ctx)
    if opts.granularity == GRANULARITY_TIMESLICE_SAMPLE:
        return _probe_timeslice_frame(ctx, opts.reference_frame)
    return False


def _load_spot(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    def _loader(sid, position_key, bdir):
        return process_position_data(
            sid,
            position_key,
            extra_input_columns=[C_ENERGY],
            base_dir=bdir,
        )

    data = try_load_position_data(
        ctx.session_id,
        ctx.base_dir,
        _loader,
        raw=spot_positions_raw(opts.reference_frame),
    )
    if data is None or "ic1_x" not in data or "ic2_x" not in data:
        return None
    out = {
        "session_id": ctx.session_id,
        "ic1_x": np.asarray(data["ic1_x"], dtype=float),
        "ic1_y": np.asarray(data["ic1_y"], dtype=float),
        "ic2_x": np.asarray(data["ic2_x"], dtype=float),
        "ic2_y": np.asarray(data["ic2_y"], dtype=float),
    }
    if "energy" in data:
        out["energy"] = np.asarray(data["energy"], dtype=float)
    return out


def _load_timeslice(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    bg_subtract = opts.resolved_bg_subtract(ctx)
    prepare, extract, keys = timeslice_position_table_hooks(opts.reference_frame)
    table = load_energy_tagged_table(
        ctx.session_id,
        ctx.base_dir,
        usecols=TIMESLICE_POSITION_ERROR_COLS,
        bg_subtract=bg_subtract,
        prepare=prepare,
        extract=extract,
        keys=keys,
    )
    if table is not None:
        return table

    loader = timeslice_position_loader(opts.reference_frame)
    positions = loader(ctx.session_id, ctx.base_dir, bg_subtract=bg_subtract)
    if positions is None:
        return None
    return {
        "session_id": ctx.session_id,
        "ic1_x": np.asarray(positions.ic1_x, dtype=float),
        "ic1_y": np.asarray(positions.ic1_y, dtype=float),
        "ic2_x": np.asarray(positions.ic2_x, dtype=float),
        "ic2_y": np.asarray(positions.ic2_y, dtype=float),
        "beam_on": positions.beam_on,
    }


def load_ic12(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    if opts.granularity == GRANULARITY_SPOT:
        return _load_spot(ctx, opts)
    if opts.granularity == GRANULARITY_TIMESLICE_SAMPLE:
        return _load_timeslice(ctx, opts)
    return None


SPEC = register(
    DataSourceSpec(
        id=SOURCE_IC12_POS_DIFF,
        label="IC2−IC1 Position",
        data_sources=frozenset({
            DATA_SOURCE_SPOT_ISO,
            DATA_SOURCE_SPOT_CHAMBER,
            DATA_SOURCE_TIMESLICE_ISO,
            DATA_SOURCE_TIMESLICE_CHAMBER,
        }),
        supports_bg_subtract=True,
        supports_beam_filter=True,
        probe=probe_ic12,
        load=load_ic12,
    )
)
