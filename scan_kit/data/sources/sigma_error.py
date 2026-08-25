"""Sigma error vs target (spot and timeslice samples)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...common import create_valid_mask, load_session_raw
from ...common.session_sigma import resolve_spot_sigma_column
from ...common.timeslice_sigma import (
    TIMESLICE_SIGMA_ERROR_COLS,
    _resolve_sigma_target_columns,
    frame_timeslice_sigma_error_arrays,
    load_session_beam_on_sigma_errors,
    resolve_timeslice_sigma_source,
    timeslice_sigma_error_available,
)
from ..context import LoadOptions, SessionContext
from ..registry import DataSourceSpec, register
from ..types import GRANULARITY_SPOT, GRANULARITY_TIMESLICE_SAMPLE, REFERENCE_ISO
from ...common.session_source import read_session_csv_columns, resolve_session_source
from ...common import detect_beam_on_mask, load_session_timeslice_device_units

SOURCE_SIGMA_ERROR = "sigma_error"

_SPOT_SIGMA_ATTRS = (
    ("ic1_x", "ic1", "x"),
    ("ic1_y", "ic1", "y"),
    ("ic2_x", "ic2", "x"),
    ("ic2_y", "ic2", "y"),
)


def _spot_sigma_target_columns(spot_cols: list[str], input_cols: list[str]) -> dict[str, str] | None:
    return _resolve_sigma_target_columns(spot_cols) or _resolve_sigma_target_columns(input_cols)


def _probe_spot(ctx: SessionContext) -> bool:
    src = resolve_session_source(ctx.session_id, ctx.base_dir)
    if src is None:
        return False
    spot_cols = read_session_csv_columns(src, "spot_data.csv")
    if not spot_cols:
        return False
    if not all(
        resolve_spot_sigma_column(spot_cols, ic, axis) is not None
        for _attr, ic, axis in _SPOT_SIGMA_ATTRS
    ):
        return False
    input_cols = read_session_csv_columns(src, "input_map.csv") or []
    return _spot_sigma_target_columns(spot_cols, input_cols) is not None


def _probe_sigma_error_timeslice(ctx: SessionContext) -> bool:
    src = resolve_session_source(ctx.session_id, ctx.base_dir)
    if src is None:
        return False
    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_SIGMA_ERROR_COLS, max_frames=1,
    )
    if not frames:
        return False
    if not timeslice_sigma_error_available(frames[0].columns):
        return False
    beam_on = detect_beam_on_mask(frames[0])
    if beam_on is None or not np.any(beam_on):
        return False
    source = resolve_timeslice_sigma_source(frames[0].columns)
    if source is None:
        return False
    target_cols = _resolve_sigma_target_columns(frames[0].columns)
    if target_cols is None:
        return False
    return frame_timeslice_sigma_error_arrays(frames[0], source, target_cols) is not None


def probe_sigma_error(ctx: SessionContext, opts: LoadOptions) -> bool:
    if opts.granularity == GRANULARITY_SPOT:
        return _probe_spot(ctx)
    if opts.granularity == GRANULARITY_TIMESLICE_SAMPLE:
        return _probe_sigma_error_timeslice(ctx)
    return False


def _load_spot(ctx: SessionContext, _opts: LoadOptions) -> dict | None:
    input_map, spot_data = load_session_raw(ctx.session_id, base_dir=ctx.base_dir)
    if spot_data is None:
        return None

    measured_cols: dict[str, str] = {}
    for attr, ic, axis in _SPOT_SIGMA_ATTRS:
        col = resolve_spot_sigma_column(spot_data.columns, ic, axis)
        if col is None:
            return None
        measured_cols[attr] = col

    input_cols = list(input_map.columns) if input_map is not None else []
    target_cols = _spot_sigma_target_columns(list(spot_data.columns), input_cols)
    if target_cols is None:
        return None

    if _resolve_sigma_target_columns(spot_data.columns) is not None:
        target_frame = spot_data[list(target_cols.values())]
    elif input_map is not None:
        target_frame = input_map[list(target_cols.values())]
    else:
        return None

    frame = spot_data[list(measured_cols.values())].copy().join(target_frame)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    clean = frame[create_valid_mask(frame)]
    if clean.empty:
        return None

    out: dict = {"session_id": ctx.session_id}
    for attr, _ic, _axis in _SPOT_SIGMA_ATTRS:
        meas = clean[measured_cols[attr]].to_numpy(dtype=float) * 2.0
        target = clean[target_cols[attr]].to_numpy(dtype=float)
        out[f"{attr}_err"] = meas - target
    return out


def _load_timeslice(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    bg_subtract = opts.resolved_bg_subtract(ctx)
    errors = load_session_beam_on_sigma_errors(
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


def load_sigma_error(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    if opts.granularity == GRANULARITY_SPOT:
        return _load_spot(ctx, opts)
    if opts.granularity == GRANULARITY_TIMESLICE_SAMPLE:
        return _load_timeslice(ctx, opts)
    return None


SPEC = register(
    DataSourceSpec(
        id=SOURCE_SIGMA_ERROR,
        label="Sigma Error",
        granularities=frozenset({GRANULARITY_SPOT, GRANULARITY_TIMESLICE_SAMPLE}),
        reference_frames=frozenset({REFERENCE_ISO}),
        supports_bg_subtract=True,
        supports_beam_filter=True,
        probe=probe_sigma_error,
        load=load_sigma_error,
    )
)
