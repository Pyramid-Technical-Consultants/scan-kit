"""Plan-relative position error (spot and timeslice samples)."""

from __future__ import annotations

import numpy as np

from ...common import (
    C_ENERGY,
    C_X_POSITION,
    C_Y_POSITION,
    process_position_data,
    try_load_position_data,
)
from ...common.timeslice_position_error import (
    TIMESLICE_POSITION_ERROR_COLS,
    frame_timeslice_error_arrays,
    load_session_beam_on_position_errors,
    resolve_session_timeslice_error_source,
)
from ...common.timeslice_table import load_energy_tagged_table
from ..context import LoadOptions, SessionContext
from ..registry import DataSourceSpec, register
from ..types import (
    GRANULARITY_SPOT,
    GRANULARITY_TIMESLICE_SAMPLE,
    REFERENCE_ISO,
)
from ._common import probe_spot_positions_with_plan, probe_timeslice_session_arrays

SOURCE_POSITION_ERROR = "position_error"

_TIMESLICE_ERROR_KEYS = ("ic1_x_err", "ic1_y_err", "ic2_x_err", "ic2_y_err")


def probe_position_error(ctx: SessionContext, opts: LoadOptions) -> bool:
    if opts.granularity == GRANULARITY_SPOT:
        return probe_spot_positions_with_plan(ctx)
    if opts.granularity == GRANULARITY_TIMESLICE_SAMPLE:
        return probe_timeslice_session_arrays(
            ctx,
            usecols=TIMESLICE_POSITION_ERROR_COLS,
            resolve_session_fn=resolve_session_timeslice_error_source,
            frame_arrays_fn=frame_timeslice_error_arrays,
        )
    return False


def _load_spot(ctx: SessionContext, _opts: LoadOptions) -> dict | None:
    def _loader(sid, position_key, bdir):
        return process_position_data(
            sid,
            position_key,
            extra_input_columns=[C_ENERGY, C_X_POSITION, C_Y_POSITION],
            base_dir=bdir,
        )

    data = try_load_position_data(ctx.session_id, ctx.base_dir, _loader, raw=False)
    if data is None:
        return None
    if C_X_POSITION not in data or C_Y_POSITION not in data:
        return None
    plan_x = np.asarray(data[C_X_POSITION], dtype=float)
    plan_y = np.asarray(data[C_Y_POSITION], dtype=float)
    out = {
        "session_id": ctx.session_id,
        "ic1_x_err": np.asarray(data["ic1_x"], dtype=float) - plan_x,
        "ic1_y_err": np.asarray(data["ic1_y"], dtype=float) - plan_y,
        "ic2_x_err": np.asarray(data["ic2_x"], dtype=float) - plan_x,
        "ic2_y_err": np.asarray(data["ic2_y"], dtype=float) - plan_y,
        "plan_x": plan_x,
        "plan_y": plan_y,
    }
    if "energy" in data:
        out["energy"] = np.asarray(data["energy"], dtype=float)
    return out


def _load_timeslice(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    bg_subtract = opts.resolved_bg_subtract(ctx)

    def prepare(src, frames):
        return resolve_session_timeslice_error_source(src, frames)

    def extract(df, error_source):
        return frame_timeslice_error_arrays(df, error_source)

    table = load_energy_tagged_table(
        ctx.session_id,
        ctx.base_dir,
        usecols=TIMESLICE_POSITION_ERROR_COLS,
        bg_subtract=bg_subtract,
        prepare=prepare,
        extract=extract,
        keys=_TIMESLICE_ERROR_KEYS,
    )
    if table is not None:
        return table

    errors = load_session_beam_on_position_errors(
        ctx.session_id, ctx.base_dir, bg_subtract=bg_subtract,
    )
    if errors is None:
        return None
    return {
        "session_id": ctx.session_id,
        "ic1_x_err": np.asarray(errors.ic1_x, dtype=float),
        "ic1_y_err": np.asarray(errors.ic1_y, dtype=float),
        "ic2_x_err": np.asarray(errors.ic2_x, dtype=float),
        "ic2_y_err": np.asarray(errors.ic2_y, dtype=float),
        "beam_on": errors.beam_on,
    }


def load_position_error(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    if opts.granularity == GRANULARITY_SPOT:
        return _load_spot(ctx, opts)
    if opts.granularity == GRANULARITY_TIMESLICE_SAMPLE:
        return _load_timeslice(ctx, opts)
    return None


SPEC = register(
    DataSourceSpec(
        id=SOURCE_POSITION_ERROR,
        label="Position Error",
        granularities=frozenset({GRANULARITY_SPOT, GRANULARITY_TIMESLICE_SAMPLE}),
        reference_frames=frozenset({REFERENCE_ISO}),
        supports_bg_subtract=True,
        supports_beam_filter=True,
        probe=probe_position_error,
        load=load_position_error,
    )
)
