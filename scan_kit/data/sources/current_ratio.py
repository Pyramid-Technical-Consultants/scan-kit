"""Energy-binned IC current ratios (timeslice)."""

from __future__ import annotations

from ...common import C_ENERGY, resolve_concept_column
from ...common.current_ratios import load_session_current_ratios
from ...common.session_source import (
    read_first_timeslice_columns,
    read_session_csv_columns,
    resolve_session_source,
)
from ...common.timeslice_ic_current import resolve_ic_current_columns
from ..context import LoadOptions, SessionContext
from ..registry import DataSourceSpec, register
from ..types import GRANULARITY_ENERGY_BINNED, REFERENCE_ISO

SOURCE_CURRENT_RATIO = "current_ratio"


def probe_current_ratio(ctx: SessionContext, opts: LoadOptions) -> bool:
    if opts.granularity != GRANULARITY_ENERGY_BINNED:
        return False
    src = resolve_session_source(ctx.session_id, ctx.base_dir)
    if src is None:
        return False
    input_cols = read_session_csv_columns(src, "input_map.csv")
    if not input_cols or resolve_concept_column(input_cols, C_ENERGY) is None:
        return False
    ts_cols = read_first_timeslice_columns(src)
    if not ts_cols:
        return False
    return resolve_ic_current_columns(ts_cols) is not None


def load_current_ratio(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    if opts.granularity != GRANULARITY_ENERGY_BINNED:
        return None
    return load_session_current_ratios(
        ctx.session_id,
        ctx.base_dir,
        bg_subtract=opts.resolved_bg_subtract(ctx),
    )


SPEC = register(
    DataSourceSpec(
        id=SOURCE_CURRENT_RATIO,
        label="Current Ratio",
        granularities=frozenset({GRANULARITY_ENERGY_BINNED}),
        reference_frames=frozenset({REFERENCE_ISO}),
        supports_bg_subtract=True,
        supports_beam_filter=False,
        probe=probe_current_ratio,
        load=load_current_ratio,
    )
)
