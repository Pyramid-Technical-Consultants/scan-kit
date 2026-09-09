"""Sigma error vs target (spot and timeslice samples)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...common import C_ENERGY, create_valid_mask, load_session_raw, resolve_concept_column
from ...common.devices_xml import IC_SIGMA_DEVICES, load_session_devices_config
from ...common.session_sigma import resolve_spot_sigma_column
from ...common.timeslice_sigma import (
    TIMESLICE_SIGMA_ERROR_COLS,
    _resolve_sigma_target_columns,
    frame_timeslice_sigma_error_arrays,
    load_session_beam_on_sigma_errors,
    resolve_timeslice_sigma_source,
    timeslice_sigma_error_available,
)
from ...common.timeslice_table import load_energy_tagged_table
from ..context import LoadOptions, SessionContext
from ..reference_frame import spot_sigma_prefer_raw
from ..registry import DataSourceSpec, register
from ..types import (
    DATA_SOURCE_SPOT_CHAMBER,
    DATA_SOURCE_SPOT_ISO,
    DATA_SOURCE_TIMESLICE_ISO,
    GRANULARITY_SPOT,
    GRANULARITY_TIMESLICE_SAMPLE,
)
from ...common.session_source import read_session_csv_columns, resolve_session_source
from ...common import detect_beam_on_mask, load_session_timeslice_device_units
from ._common import beam_on_has_samples

SOURCE_SIGMA_ERROR = "sigma_error"

_TIMESLICE_SIGMA_ERROR_KEYS = ("ic1_x_err", "ic1_y_err", "ic2_x_err", "ic2_y_err")

_SPOT_SIGMA_ATTRS = (
    ("ic1_x", "ic1", "x"),
    ("ic1_y", "ic1", "y"),
    ("ic2_x", "ic2", "x"),
    ("ic2_y", "ic2", "y"),
)

_SPOT_ATTR_TO_DEVICE = {
    "ic1_x": "IC_1_X",
    "ic1_y": "IC_1_Y",
    "ic2_x": "IC_2_X",
    "ic2_y": "IC_2_Y",
}


def _spot_sigma_target_columns(spot_cols: list[str], input_cols: list[str]) -> dict[str, str] | None:
    return _resolve_sigma_target_columns(spot_cols) or _resolve_sigma_target_columns(input_cols)


def _resolve_spot_measured_columns(
    spot_cols: list[str],
    *,
    prefer_raw: bool,
) -> dict[str, str] | None:
    measured_cols: dict[str, str] = {}
    for attr, ic, axis in _SPOT_SIGMA_ATTRS:
        col = resolve_spot_sigma_column(
            spot_cols, ic, axis, prefer_raw=prefer_raw,
        )
        if col is None:
            return None
        measured_cols[attr] = col
    return measured_cols


def _spot_can_use_devices_xml_targets(
    ctx: SessionContext,
    input_cols: list[str],
) -> bool:
    if resolve_concept_column(input_cols, C_ENERGY) is None:
        return False
    config = load_session_devices_config(ctx.session_id, ctx.base_dir)
    if config is None:
        return False
    return any(config.beam_sigmas.get(device) for device in IC_SIGMA_DEVICES)


def _expected_sigma_targets(
    config,
    device: str,
    energies: np.ndarray,
) -> np.ndarray:
    targets = np.empty(len(energies), dtype=float)
    for i, energy in enumerate(energies):
        if not np.isfinite(energy):
            targets[i] = np.nan
            continue
        expected = config.expected_sigma_mm(device, float(energy))
        targets[i] = float(expected) if expected is not None else np.nan
    return targets


def _probe_spot(ctx: SessionContext, opts: LoadOptions) -> bool:
    src = resolve_session_source(ctx.session_id, ctx.base_dir)
    if src is None:
        return False
    spot_cols = read_session_csv_columns(src, "spot_data.csv")
    if not spot_cols:
        return False
    prefer_raw = spot_sigma_prefer_raw(opts.reference_frame)
    if _resolve_spot_measured_columns(spot_cols, prefer_raw=prefer_raw) is None:
        return False
    input_cols = read_session_csv_columns(src, "input_map.csv") or []
    if _spot_sigma_target_columns(spot_cols, input_cols) is not None:
        return True
    return _spot_can_use_devices_xml_targets(ctx, input_cols)


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
    if not beam_on_has_samples(beam_on):
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
        return _probe_spot(ctx, opts)
    if opts.granularity == GRANULARITY_TIMESLICE_SAMPLE:
        return _probe_sigma_error_timeslice(ctx)
    return False


def _load_spot(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    input_map, spot_data = load_session_raw(ctx.session_id, base_dir=ctx.base_dir)
    if spot_data is None:
        return None

    prefer_raw = spot_sigma_prefer_raw(opts.reference_frame)
    measured_cols = _resolve_spot_measured_columns(
        list(spot_data.columns),
        prefer_raw=prefer_raw,
    )
    if measured_cols is None:
        return None

    input_cols = list(input_map.columns) if input_map is not None else []
    target_cols = _spot_sigma_target_columns(list(spot_data.columns), input_cols)
    use_devices_xml = target_cols is None
    devices_config = None
    energy_col = None

    if use_devices_xml:
        if input_map is None:
            return None
        energy_col = resolve_concept_column(input_map.columns, C_ENERGY)
        if energy_col is None:
            return None
        devices_config = load_session_devices_config(ctx.session_id, ctx.base_dir)
        if devices_config is None:
            return None
        frame = spot_data[list(measured_cols.values())].copy().join(input_map[energy_col])
    else:
        if _resolve_sigma_target_columns(spot_data.columns) is not None:
            target_frame = spot_data[list(target_cols.values())]
        elif input_map is not None:
            target_frame = input_map[list(target_cols.values())]
        else:
            return None
        frame = spot_data[list(measured_cols.values())].copy().join(target_frame)
        if input_map is not None:
            energy_col = resolve_concept_column(input_map.columns, C_ENERGY)
            if energy_col is not None:
                frame = frame.join(input_map[energy_col])

    frame = frame.apply(pd.to_numeric, errors="coerce")
    clean = frame[create_valid_mask(frame)]
    if clean.empty:
        return None

    out: dict = {"session_id": ctx.session_id}
    if energy_col is not None:
        out["energy"] = clean[energy_col].to_numpy(dtype=float)

    energies = out.get("energy")
    err_keys: list[str] = []
    for attr, _ic, _axis in _SPOT_SIGMA_ATTRS:
        meas = clean[measured_cols[attr]].to_numpy(dtype=float) * 2.0
        if use_devices_xml:
            if energies is None:
                return None
            device = _SPOT_ATTR_TO_DEVICE[attr]
            target = _expected_sigma_targets(devices_config, device, energies)
        else:
            target = clean[target_cols[attr]].to_numpy(dtype=float)
        key = f"{attr}_err"
        out[key] = meas - target
        err_keys.append(key)

    if not any(np.any(np.isfinite(out[key])) for key in err_keys):
        return None
    return out


def _load_timeslice(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    bg_subtract = opts.resolved_bg_subtract(ctx)

    def prepare(_src, frames):
        return resolve_timeslice_sigma_source(frames[0].columns)

    def extract(df, source):
        target_cols = _resolve_sigma_target_columns(df.columns)
        if target_cols is None:
            return None
        return frame_timeslice_sigma_error_arrays(df, source, target_cols)

    table = load_energy_tagged_table(
        ctx.session_id,
        ctx.base_dir,
        usecols=TIMESLICE_SIGMA_ERROR_COLS,
        bg_subtract=bg_subtract,
        prepare=prepare,
        extract=extract,
        keys=_TIMESLICE_SIGMA_ERROR_KEYS,
    )
    if table is not None:
        return table

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
        data_sources=frozenset({
            DATA_SOURCE_SPOT_ISO,
            DATA_SOURCE_SPOT_CHAMBER,
            DATA_SOURCE_TIMESLICE_ISO,
        }),
        supports_bg_subtract=True,
        supports_beam_filter=True,
        probe=probe_sigma_error,
        load=load_sigma_error,
    )
)
