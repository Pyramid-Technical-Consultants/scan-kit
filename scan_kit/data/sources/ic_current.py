"""Energy-binned IC beam current (timeslice)."""

from __future__ import annotations

from ...common.ic_current_timeslice import load_session_ic_current_timeslice
from ...common.session_source import read_first_timeslice_columns, resolve_session_source
from ...common.timeslice_ic_current import resolve_ic_current_columns
from ..context import LoadOptions, SessionContext
from ..registry import DataSourceSpec, register
from ..types import (
    DATA_SOURCE_TIMESLICE_ISO,
    GRANULARITY_ENERGY_BINNED,
)

SOURCE_IC_CURRENT = "ic_current"


def probe_ic_current(ctx: SessionContext, opts: LoadOptions) -> bool:
    if opts.granularity != GRANULARITY_ENERGY_BINNED:
        return False
    src = resolve_session_source(ctx.session_id, ctx.base_dir)
    if src is None:
        return False
    ts_cols = read_first_timeslice_columns(src)
    if not ts_cols:
        return False
    return resolve_ic_current_columns(ts_cols) is not None


def load_ic_current(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    if opts.granularity != GRANULARITY_ENERGY_BINNED:
        return None
    return load_session_ic_current_timeslice(
        ctx.session_id,
        ctx.base_dir,
        bg_subtract=opts.resolved_bg_subtract(ctx),
    )


SPEC = register(
    DataSourceSpec(
        id=SOURCE_IC_CURRENT,
        label="IC Current",
        data_sources=frozenset({DATA_SOURCE_TIMESLICE_ISO}),
        granularity_for={DATA_SOURCE_TIMESLICE_ISO: GRANULARITY_ENERGY_BINNED},
        supports_bg_subtract=True,
        supports_beam_filter=False,
        probe=probe_ic_current,
        load=load_ic_current,
    )
)
