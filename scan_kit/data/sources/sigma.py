"""IC sigma (spot and timeslice samples)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...common import C_ENERGY, create_valid_mask, load_session_raw, resolve_concept_column
from ...common.session_sigma import IC_SIGMA_LABELS, resolve_spot_sigma_column
from ...common.timeslice_sigma import (
    TIMESLICE_SIGMA_COLS,
    frame_timeslice_sigma_arrays,
    load_session_beam_on_sigmas,
    resolve_timeslice_sigma_source,
)
from ...common.timeslice_table import load_energy_tagged_table
from ..context import LoadOptions, SessionContext
from ..reference_frame import spot_sigma_prefer_raw
from ..registry import DataSourceSpec, register
from ..types import (
    DATA_SOURCE_SPOT_ISO,
    DATA_SOURCE_SPOT_CHAMBER,
    DATA_SOURCE_TIMESLICE_ISO,
    GRANULARITY_SPOT,
    GRANULARITY_TIMESLICE_SAMPLE,
)
from ._common import probe_timeslice_frame
from ...common.session_source import read_session_csv_columns, resolve_session_source

SOURCE_SIGMA = "sigma"

_SIGMA_KEYS = ("ic1_sig_x", "ic1_sig_y", "ic2_sig_x", "ic2_sig_y")


def _probe_spot(ctx: SessionContext, prefer_raw: bool | None) -> bool:
    src = resolve_session_source(ctx.session_id, ctx.base_dir)
    if src is None:
        return False
    spot_cols = read_session_csv_columns(src, "spot_data.csv")
    if not spot_cols:
        return False
    return all(
        resolve_spot_sigma_column(spot_cols, ic, axis, prefer_raw=prefer_raw) is not None
        for _label, ic, axis in IC_SIGMA_LABELS
    )


def probe_sigma(ctx: SessionContext, opts: LoadOptions) -> bool:
    if opts.granularity == GRANULARITY_SPOT:
        return _probe_spot(ctx, spot_sigma_prefer_raw(opts.reference_frame))
    if opts.granularity == GRANULARITY_TIMESLICE_SAMPLE:
        return probe_timeslice_frame(
            ctx,
            usecols=TIMESLICE_SIGMA_COLS,
            resolve_source=resolve_timeslice_sigma_source,
            frame_arrays=frame_timeslice_sigma_arrays,
        )
    return False


def _load_spot(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    prefer_raw = spot_sigma_prefer_raw(opts.reference_frame)
    input_map, spot_data = load_session_raw(ctx.session_id, base_dir=ctx.base_dir)
    if input_map is None or spot_data is None:
        return None
    energy_col = resolve_concept_column(input_map.columns, C_ENERGY)
    if energy_col is None:
        return None

    found: dict[str, str] = {}
    for label, ic, axis in IC_SIGMA_LABELS:
        col = resolve_spot_sigma_column(
            spot_data.columns, ic, axis, prefer_raw=prefer_raw,
        )
        if col is not None:
            found[label] = col
    if not found:
        return None

    merged = spot_data[list(found.values())].copy().join(input_map[energy_col])
    merged = merged.apply(pd.to_numeric, errors="coerce")
    clean = merged[create_valid_mask(merged)]
    if clean.empty:
        return None
    out: dict = {
        "session_id": ctx.session_id,
        "energy": clean[energy_col].values.astype(float),
    }
    for label, raw_col in found.items():
        out[label] = clean[raw_col].values.astype(float) * 2.0
    return out


def _load_timeslice(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    bg_subtract = opts.resolved_bg_subtract(ctx)

    def prepare(_src, frames):
        return resolve_timeslice_sigma_source(frames[0].columns)

    def extract(df, source):
        return frame_timeslice_sigma_arrays(df, source)

    table = load_energy_tagged_table(
        ctx.session_id,
        ctx.base_dir,
        usecols=TIMESLICE_SIGMA_COLS,
        bg_subtract=bg_subtract,
        prepare=prepare,
        extract=extract,
        keys=_SIGMA_KEYS,
    )
    if table is not None:
        return table

    sigmas = load_session_beam_on_sigmas(
        ctx.session_id, ctx.base_dir, bg_subtract=bg_subtract,
    )
    if sigmas is None:
        return None
    return {
        "session_id": ctx.session_id,
        "ic1_sig_x": np.asarray(sigmas.ic1_x, dtype=float),
        "ic1_sig_y": np.asarray(sigmas.ic1_y, dtype=float),
        "ic2_sig_x": np.asarray(sigmas.ic2_x, dtype=float),
        "ic2_sig_y": np.asarray(sigmas.ic2_y, dtype=float),
        "beam_on": sigmas.beam_on,
    }


def load_sigma(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    if opts.granularity == GRANULARITY_SPOT:
        return _load_spot(ctx, opts)
    if opts.granularity == GRANULARITY_TIMESLICE_SAMPLE:
        return _load_timeslice(ctx, opts)
    return None


SPEC = register(
    DataSourceSpec(
        id=SOURCE_SIGMA,
        label="Sigma",
        data_sources=frozenset({
            DATA_SOURCE_SPOT_ISO,
            DATA_SOURCE_SPOT_CHAMBER,
            DATA_SOURCE_TIMESLICE_ISO,
        }),
        supports_bg_subtract=True,
        supports_beam_filter=True,
        probe=probe_sigma,
        load=load_sigma,
    )
)
