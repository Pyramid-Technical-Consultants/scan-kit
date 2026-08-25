"""Unified per-spot summary table loader for the binned summary viewer."""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..common import (
    C_CHARGE_REQ,
    C_ENERGY,
    C_IC1_TOTAL_DOSE,
    C_IC2_TOTAL_DOSE,
    C_IC3_TOTAL_DOSE,
    C_X_POSITION,
    C_Y_POSITION,
    DELIVERED_DOSE_COLS,
    ViewSettings,
    add_dose_error_columns,
    add_dose_ratio_columns,
    apply_auto_calibration,
    apply_calibration_factors,
    create_valid_mask,
    load_session_raw,
    process_position_data,
    resolve_concept_column,
    try_load_position_data,
)
from ..common.timeslice_table import load_energy_tagged_table
from ..common.timeslice_position_error import (
    TIMESLICE_POSITION_ERROR_COLS,
    frame_timeslice_error_arrays,
    resolve_session_timeslice_error_source,
)
from ..common.timeslice_sigma import (
    TIMESLICE_SIGMA_COLS,
    frame_timeslice_sigma_arrays,
    resolve_timeslice_sigma_source,
)
from ..common.current_ratios import load_session_current_ratios
from ..common.ic_current_timeslice import load_session_ic_current_timeslice
from ..common.mu_delivery_rate import load_session_mu_delivery_rates
from .binned_summary_catalog import (
    DATA_SOURCE_SPOT,
    DATA_SOURCE_TIMESLICE,
    GLYPH_VIOLIN,
    X_ENERGY,
    X_PARAMS,
    VIEW_OPTIONS,
    Y_DOSE_RATE,
    Y_CURRENT_RATIO,
    Y_IC_CURRENT,
    Y_GROUP_BY_ID,
    Y_GROUPS,
    BinnedSummaryConfig,
)
from ..common.session_sigma import IC_SIGMA_LABELS, resolve_spot_sigma_column
from .distribution_data import probe_session_for_mode
from .distribution_catalog import (
    MODE_POSITION_ERROR_TIMESLICE,
    MODE_SIGMA_TIMESLICE,
)
from .unified_catalog import DataSourceKind, is_option_available, option_key

_log = logging.getLogger(__name__)

_EXTRA_SPOT = [
    C_IC1_TOTAL_DOSE,
    C_IC2_TOTAL_DOSE,
    C_IC3_TOTAL_DOSE,
    "timestamp",
    "layer_id",
    "spot_no",
    "time_s",
    "time_ns",
]


def _ensure_timestamp(data: dict) -> dict:
    if "timestamp" in data:
        return data
    if "time_s" not in data or "time_ns" not in data:
        return data
    result = dict(data)
    result["timestamp"] = (
        np.asarray(result["time_s"], dtype=float) * 1000.0
        + np.asarray(result["time_ns"], dtype=float) / 1e6
    )
    return result


def _load_sigma_columns(session_id: str, base_dir: str) -> dict | None:
    input_map, spot_data = load_session_raw(session_id, base_dir=base_dir)
    if input_map is None or spot_data is None:
        return None
    energy_col = resolve_concept_column(input_map.columns, C_ENERGY)
    if energy_col is None:
        return None

    found: dict[str, str] = {}
    for label, ic, axis in IC_SIGMA_LABELS:
        col = resolve_spot_sigma_column(spot_data.columns, ic, axis)
        if col is not None:
            found[label] = col
    if not found:
        return None

    merged = spot_data[list(found.values())].copy().join(input_map[energy_col])
    merged = merged.apply(pd.to_numeric, errors="coerce")
    clean = merged[create_valid_mask(merged)]
    if clean.empty:
        return None
    out = {"energy": clean[energy_col].values.astype(float)}
    for label, raw_col in found.items():
        out[label] = clean[raw_col].values.astype(float) * 2.0
    return out


def _load_position_errors(session_id: str, base_dir: str) -> dict | None:
    def _loader(sid, position_key, bdir):
        return process_position_data(
            sid,
            position_key,
            extra_input_columns=[C_X_POSITION, C_Y_POSITION],
            base_dir=bdir,
        )

    data = try_load_position_data(session_id, base_dir, _loader, raw=False)
    if data is None:
        return None
    if C_X_POSITION not in data or C_Y_POSITION not in data:
        return None
    plan_x = np.asarray(data[C_X_POSITION], dtype=float)
    plan_y = np.asarray(data[C_Y_POSITION], dtype=float)
    out = {
        "energy": np.asarray(data["energy"], dtype=float),
        "ic1_x_err": np.asarray(data["ic1_x"], dtype=float) - plan_x,
        "ic1_y_err": np.asarray(data["ic1_y"], dtype=float) - plan_y,
        "ic2_x_err": np.asarray(data["ic2_x"], dtype=float) - plan_x,
        "ic2_y_err": np.asarray(data["ic2_y"], dtype=float) - plan_y,
        "plan_x": plan_x,
        "plan_y": plan_y,
    }
    return out


def _align_by_length(base: dict, extra: dict, keys: Sequence[str]) -> None:
    """Copy *keys* from *extra* into *base* when row counts match."""
    n = len(np.asarray(base.get("energy", [])))
    if n == 0:
        return
    e_extra = np.asarray(extra.get("energy", []), dtype=float)
    if len(e_extra) != n:
        return
    for key in keys:
        if key in extra and key != "energy":
            base[key] = np.asarray(extra[key], dtype=float)


def load_session_summary_table(
    session_id: str,
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
) -> dict | None:
    """Load one session into a fat per-spot column bag."""

    def _loader(sid, position_key, bdir):
        data = process_position_data(
            sid,
            position_key,
            extra_spot_columns=_EXTRA_SPOT,
            extra_input_columns=[C_CHARGE_REQ, C_X_POSITION, C_Y_POSITION],
            base_dir=bdir,
        )
        if data is None:
            return None
        data = dict(data)
        dose_cols = [
            c for c in (C_IC1_TOTAL_DOSE, C_IC2_TOTAL_DOSE, C_IC3_TOTAL_DOSE)
            if c in data
        ]
        if settings and settings.auto_calibrate and dose_cols and C_CHARGE_REQ in data:
            if settings.cal_factors:
                data = apply_calibration_factors(data, dose_cols, settings.cal_factors)
            else:
                data = apply_auto_calibration(data, C_CHARGE_REQ, dose_cols)
        include_ic3 = C_IC3_TOTAL_DOSE in data
        ratios = add_dose_ratio_columns(data, include_ic3=include_ic3)
        if ratios is not None:
            data = ratios
        errors = add_dose_error_columns(
            data, target_col=C_CHARGE_REQ, delivered_cols=DELIVERED_DOSE_COLS,
        )
        if errors is not None:
            data = errors
        return data

    data = try_load_position_data(session_id, base_dir, _loader, raw=True)
    if data is None:
        # Fall back to a minimal energy-only bag so sigma-only sessions still work.
        data = {"session_id": session_id}

    result = dict(data)
    if "energy" in result:
        result["energy"] = np.asarray(result["energy"], dtype=float)

    if C_CHARGE_REQ in result:
        result["target_mu"] = np.asarray(result[C_CHARGE_REQ], dtype=float)

    # Radius from plan position when available, else IC1.
    if C_X_POSITION in result and C_Y_POSITION in result:
        px = np.asarray(result[C_X_POSITION], dtype=float)
        py = np.asarray(result[C_Y_POSITION], dtype=float)
        result["radius"] = np.hypot(px, py)
    elif "ic1_x" in result and "ic1_y" in result:
        result["radius"] = np.hypot(
            np.asarray(result["ic1_x"], dtype=float),
            np.asarray(result["ic1_y"], dtype=float),
        )

    result = _ensure_timestamp(result)
    # Compute spot_time without row filtering so other merged metrics stay aligned.
    if "timestamp" in result and "layer_id" in result:
        df_t = pd.DataFrame(
            {
                "timestamp": np.asarray(result["timestamp"], dtype=float),
                "layer_id": np.asarray(result["layer_id"]),
            }
        )
        spot_time = df_t.groupby("layer_id")["timestamp"].diff()
        first_mask = spot_time.isna()
        spot_time.loc[first_mask] = df_t.loc[first_mask, "timestamp"]
        result["spot_time"] = spot_time.to_numpy(dtype=float)

    pos_err = _load_position_errors(session_id, base_dir)
    if pos_err is not None:
        if "energy" not in result:
            result["energy"] = pos_err["energy"]
        _align_by_length(
            result, pos_err,
            ("ic1_x_err", "ic1_y_err", "ic2_x_err", "ic2_y_err", "plan_x", "plan_y"),
        )
        if "radius" not in result and "plan_x" in pos_err:
            result["radius"] = np.hypot(pos_err["plan_x"], pos_err["plan_y"])

    sigma = _load_sigma_columns(session_id, base_dir)
    if sigma is not None:
        if "energy" not in result:
            result.update(sigma)
            result["session_id"] = session_id
        else:
            _align_by_length(
                result, sigma,
                ("ic1_sig_x", "ic1_sig_y", "ic2_sig_x", "ic2_sig_y"),
            )

    if "energy" not in result:
        return None
    n = len(np.asarray(result["energy"]))
    if n == 0:
        return None
    result["session_id"] = session_id
    return result


def load_session_dose_rate_table(session_id: str, base_dir: str) -> dict | None:
    """Load one session as layer-level dose rate vs energy rows."""
    rates = load_session_mu_delivery_rates(session_id, base_dir)
    if rates is None:
        return None
    return {
        "session_id": session_id,
        "energy": rates["energy"],
        "mu_rate": rates["mu_rate"],
        "session_avg_rate": rates["session_avg_rate"],
    }


def _load_sessions_map(
    session_ids: Sequence[str],
    loader: Callable[[str], dict | None],
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sid in session_ids:
        data = loader(sid)
        if data is not None:
            out[sid] = data
    return out


def load_sessions_dose_rate(
    session_ids: Sequence[str],
    base_dir: str,
) -> dict[str, dict]:
    return _load_sessions_map(
        session_ids,
        lambda sid: load_session_dose_rate_table(sid, base_dir),
    )


def load_sessions_current_ratios(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
) -> dict[str, dict]:
    bg = settings.bg_subtract if settings else False
    return _load_sessions_map(
        session_ids,
        lambda sid: load_session_current_ratios(sid, base_dir, bg_subtract=bg),
    )


def load_sessions_ic_current(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
) -> dict[str, dict]:
    bg = settings.bg_subtract if settings else False
    return _load_sessions_map(
        session_ids,
        lambda sid: load_session_ic_current_timeslice(sid, base_dir, bg_subtract=bg),
    )


def load_sessions_summary(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
) -> dict[str, dict]:
    return _load_sessions_map(
        session_ids,
        lambda sid: load_session_summary_table(sid, base_dir, settings=settings),
    )


def _load_timeslice_position_errors(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
) -> dict | None:
    def prepare(src, frames):
        return resolve_session_timeslice_error_source(src, frames)

    def extract(df, error_source):
        return frame_timeslice_error_arrays(df, error_source)

    table = load_energy_tagged_table(
        session_id,
        base_dir,
        usecols=TIMESLICE_POSITION_ERROR_COLS,
        bg_subtract=bg_subtract,
        prepare=prepare,
        extract=extract,
        keys=("ic1_x_err", "ic1_y_err", "ic2_x_err", "ic2_y_err"),
    )
    return table


def _load_timeslice_sigmas(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
) -> dict | None:
    def prepare(_src, frames):
        return resolve_timeslice_sigma_source(frames[0].columns)

    def extract(df, source):
        return frame_timeslice_sigma_arrays(df, source)

    return load_energy_tagged_table(
        session_id,
        base_dir,
        usecols=TIMESLICE_SIGMA_COLS,
        bg_subtract=bg_subtract,
        prepare=prepare,
        extract=extract,
        keys=("ic1_sig_x", "ic1_sig_y", "ic2_sig_x", "ic2_sig_y"),
    )


def load_session_timeslice_summary_table(
    session_id: str,
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
) -> dict | None:
    """Load beam-on timeslice samples binned by energy for summary plots."""
    bg_subtract = settings.bg_subtract if settings else False
    pos_err = _load_timeslice_position_errors(
        session_id, base_dir, bg_subtract=bg_subtract,
    )
    sigma = _load_timeslice_sigmas(session_id, base_dir, bg_subtract=bg_subtract)

    if pos_err is None and sigma is None:
        return None

    result: dict = {"session_id": session_id}
    if pos_err is not None:
        result.update(pos_err)
    if sigma is not None:
        if "energy" not in result:
            result.update(sigma)
        else:
            _align_by_length(
                result,
                sigma,
                ("ic1_sig_x", "ic1_sig_y", "ic2_sig_x", "ic2_sig_y"),
            )
    if "energy" not in result:
        return None
    return result


def load_sessions_timeslice_summary(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
) -> dict[str, dict]:
    return _load_sessions_map(
        session_ids,
        lambda sid: load_session_timeslice_summary_table(
            sid, base_dir, settings=settings,
        ),
    )


def load_sessions_for_source(
    session_ids: Sequence[str],
    base_dir: str,
    source: DataSourceKind,
    *,
    settings: ViewSettings | None = None,
) -> dict[str, dict]:
    if source == DATA_SOURCE_TIMESLICE:
        return load_sessions_timeslice_summary(
            session_ids, base_dir, settings=settings,
        )
    return load_sessions_summary(session_ids, base_dir, settings=settings)


def _probe_timeslice_y_group(
    session_ids: Sequence[str],
    base_dir: str,
    y_group: str,
    *,
    settings: ViewSettings | None = None,
) -> bool:
    from .binned_summary_catalog import Y_CURRENT_RATIO, Y_IC_CURRENT, Y_POSITION_ERROR, Y_SIGMA

    if y_group == Y_CURRENT_RATIO:
        return bool(load_sessions_current_ratios(session_ids, base_dir, settings=settings))
    if y_group == Y_IC_CURRENT:
        return bool(load_sessions_ic_current(session_ids, base_dir, settings=settings))

    mode_map = {
        Y_POSITION_ERROR: MODE_POSITION_ERROR_TIMESLICE,
        Y_SIGMA: MODE_SIGMA_TIMESLICE,
    }
    mode = mode_map.get(y_group)
    if mode is None:
        return False
    return any(
        probe_session_for_mode(sid, mode, base_dir) for sid in session_ids
    )


def probe_view_option_availability(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    spot_data: dict[str, dict] | None = None,
    dose_rate_data: dict[str, dict] | None = None,
    current_ratio_data: dict[str, dict] | None = None,
    ic_current_data: dict[str, dict] | None = None,
    settings: ViewSettings | None = None,
) -> dict[str, bool]:
    """Return availability for every unified binned-summary option."""
    spot = spot_data if spot_data is not None else load_sessions_summary(
        session_ids, base_dir,
    )
    dose_rate = dose_rate_data if dose_rate_data is not None else load_sessions_dose_rate(
        session_ids, base_dir,
    )
    current_ratio = (
        current_ratio_data
        if current_ratio_data is not None
        else load_sessions_current_ratios(session_ids, base_dir, settings=settings)
    )
    ic_current = (
        ic_current_data
        if ic_current_data is not None
        else load_sessions_ic_current(session_ids, base_dir, settings=settings)
    )
    spot_y = available_y_groups(spot)
    if Y_DOSE_RATE in available_y_groups(dose_rate):
        spot_y.add(Y_DOSE_RATE)
    availability: dict[str, bool] = {}
    for opt in VIEW_OPTIONS:
        key = option_key(opt.source, opt.id)
        if opt.source == DATA_SOURCE_SPOT:
            availability[key] = opt.id in spot_y
        else:
            if opt.id == Y_CURRENT_RATIO and current_ratio_data is not None:
                availability[key] = opt.id in available_y_groups(current_ratio_data)
            elif opt.id == Y_IC_CURRENT and ic_current_data is not None:
                availability[key] = opt.id in available_y_groups(ic_current_data)
            else:
                availability[key] = _probe_timeslice_y_group(
                    session_ids, base_dir, opt.id, settings=settings,
                )
    return availability


def available_x_params_for_source(
    session_data: dict[str, dict],
    source: DataSourceKind,
) -> set[str]:
    if source == DATA_SOURCE_TIMESLICE:
        if any(
            "energy" in data
            and np.isfinite(np.asarray(data["energy"], dtype=float)).any()
            for data in session_data.values()
        ):
            return {X_ENERGY}
        return set()
    return available_x_params(session_data)


def default_config(
    session_data: dict[str, dict],
    *,
    source: DataSourceKind = DATA_SOURCE_SPOT,
    option_availability: dict[str, bool] | None = None,
) -> BinnedSummaryConfig:
    if option_availability is not None:
        y_group = next(
            (
                opt.id
                for opt in VIEW_OPTIONS
                if opt.source == source
                and is_option_available(option_availability, opt)
            ),
            Y_GROUPS[0].id,
        )
    else:
        y_avail = available_y_groups(session_data)
        y_group = next((g.id for g in Y_GROUPS if g.id in y_avail), Y_GROUPS[0].id)
    x_avail = available_x_params_for_source(session_data, source)
    if x_avail:
        x_param = next((x.id for x in X_PARAMS if x.id in x_avail), next(iter(x_avail)))
    else:
        x_param = X_ENERGY
    glyph = GLYPH_VIOLIN
    return BinnedSummaryConfig(y_group=y_group, source=source, x_param=x_param, glyph=glyph)


def available_y_groups(session_data: dict[str, dict]) -> set[str]:
    available: set[str] = set()
    for group in Y_GROUPS:
        if any(
            any(series.key in data for series in group.series)
            for data in session_data.values()
        ):
            available.add(group.id)
    return available


def available_x_params(session_data: dict[str, dict]) -> set[str]:
    available: set[str] = set()
    for param in X_PARAMS:
        if any(
            param.column in data
            and np.isfinite(np.asarray(data[param.column], dtype=float)).any()
            for data in session_data.values()
        ):
            available.add(param.id)
    return available


def available_series_keys(session_data: dict[str, dict], y_group: str) -> list[str]:
    group = Y_GROUP_BY_ID.get(y_group)
    if group is None:
        return []
    keys = []
    for series in group.series:
        if any(series.key in data for data in session_data.values()):
            keys.append(series.key)
    return keys
