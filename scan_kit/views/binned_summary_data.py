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
from .binned_summary_catalog import (
    DATA_SOURCE_SPOT,
    DATA_SOURCE_TIMESLICE,
    GLYPH_VIOLIN,
    REFERENCE_ISO,
    X_ENERGY,
    X_PARAMS,
    VIEW_OPTIONS,
    Y_DOSE_RATE,
    Y_CURRENT_RATIO,
    Y_IC_CURRENT,
    Y_IC12_POS_DIFF,
    Y_POSITION_ERROR,
    Y_SIGMA,
    Y_GROUP_BY_ID,
    Y_GROUPS,
    BinnedSummaryConfig,
)
from ..data.adapters.binned import (
    ic12_to_binned_columns,
    position_error_to_binned_columns,
    sigma_to_binned_columns,
)
from ..data.context import LoadOptions, SessionContext
from ..data.registry import load as load_source
from ..data.sources.current_ratio import SOURCE_CURRENT_RATIO
from ..data.sources.dose_rate import SOURCE_DOSE_RATE
from ..data.sources.ic12_pos_diff import SOURCE_IC12_POS_DIFF
from ..data.sources.ic_current import SOURCE_IC_CURRENT
from ..data.sources.position_error import SOURCE_POSITION_ERROR
from ..data.sources.sigma import SOURCE_SIGMA
from ..data.types import (
    GRANULARITY_ENERGY_BINNED,
    GRANULARITY_LAYER,
    GRANULARITY_SPOT,
    GRANULARITY_TIMESLICE_SAMPLE,
)
from .unified_catalog import DataSourceKind, ReferenceFrameKind, is_option_available, option_key

_log = logging.getLogger(__name__)

_REGISTRY_Y_GROUPS = frozenset({
    Y_POSITION_ERROR,
    Y_SIGMA,
    Y_IC12_POS_DIFF,
    Y_DOSE_RATE,
    Y_CURRENT_RATIO,
    Y_IC_CURRENT,
})

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


def _load_registry_columns(
    source_id: str,
    session_id: str,
    base_dir: str,
    opts: LoadOptions,
    adapter: Callable[[dict | None], dict | None],
) -> dict | None:
    payload = load_source(
        source_id,
        SessionContext(session_id, base_dir),
        opts,
    )
    return adapter(payload)


def _load_sigma_columns(
    session_id: str,
    base_dir: str,
    *,
    prefer_raw: bool | None = None,
    reference_frame: ReferenceFrameKind = REFERENCE_ISO,
) -> dict | None:
    del prefer_raw  # reference_frame selects column preference in registry
    return _load_registry_columns(
        SOURCE_SIGMA,
        session_id,
        base_dir,
        LoadOptions(granularity=GRANULARITY_SPOT, reference_frame=reference_frame),
        sigma_to_binned_columns,
    )


def _load_spot_ic12_diff_columns(
    session_id: str,
    base_dir: str,
    *,
    reference_frame: ReferenceFrameKind,
) -> dict | None:
    return _load_registry_columns(
        SOURCE_IC12_POS_DIFF,
        session_id,
        base_dir,
        LoadOptions(granularity=GRANULARITY_SPOT, reference_frame=reference_frame),
        ic12_to_binned_columns,
    )


def _load_position_errors(session_id: str, base_dir: str) -> dict | None:
    return _load_registry_columns(
        SOURCE_POSITION_ERROR,
        session_id,
        base_dir,
        LoadOptions(granularity=GRANULARITY_SPOT),
        position_error_to_binned_columns,
    )


def _align_by_length(base: dict, extra: dict, keys: Sequence[str]) -> bool:
    """Copy *keys* from *extra* into *base* when row counts match."""
    n = len(np.asarray(base.get("energy", [])))
    if n == 0:
        return False
    e_extra = np.asarray(extra.get("energy", []), dtype=float)
    if len(e_extra) != n:
        return False
    for key in keys:
        if key in extra and key != "energy":
            base[key] = np.asarray(extra[key], dtype=float)
    return True


def load_session_summary_table(
    session_id: str,
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
    reference_frame: ReferenceFrameKind = REFERENCE_ISO,
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

    sigma = _load_sigma_columns(
        session_id,
        base_dir,
        reference_frame=reference_frame,
    )
    if sigma is not None:
        if "energy" not in result:
            result.update(sigma)
            result["session_id"] = session_id
        else:
            _align_by_length(
                result, sigma,
                ("ic1_sig_x", "ic1_sig_y", "ic2_sig_x", "ic2_sig_y"),
            )

    ic12 = _load_spot_ic12_diff_columns(
        session_id, base_dir, reference_frame=reference_frame,
    )
    if ic12 is not None:
        if "energy" not in result:
            result.update(ic12)
            result["session_id"] = session_id
        elif not _align_by_length(result, ic12, ("ic12_x_diff", "ic12_y_diff")):
            _log.debug(
                "Session %s: ic12 diff row count (%d) differs from spot table (%d); "
                "reference=%s",
                session_id,
                len(np.asarray(ic12.get("energy", []))),
                len(np.asarray(result.get("energy", []))),
                reference_frame,
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
    return load_source(
        SOURCE_DOSE_RATE,
        SessionContext(session_id, base_dir),
        LoadOptions(granularity=GRANULARITY_LAYER),
    )


def _load_sessions_registry(
    source_id: str,
    granularity: str,
    session_ids: Sequence[str],
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
) -> dict[str, dict]:
    bg = settings.bg_subtract if settings else False
    return _load_sessions_map(
        session_ids,
        lambda sid: load_source(
            source_id,
            SessionContext(sid, base_dir, settings),
            LoadOptions(granularity=granularity, bg_subtract=bg),
        ),
    )


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
    return _load_sessions_map(session_ids, lambda sid: load_session_dose_rate_table(sid, base_dir))


def load_sessions_current_ratios(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
) -> dict[str, dict]:
    return _load_sessions_registry(
        SOURCE_CURRENT_RATIO,
        GRANULARITY_ENERGY_BINNED,
        session_ids,
        base_dir,
        settings=settings,
    )


def load_sessions_ic_current(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
) -> dict[str, dict]:
    return _load_sessions_registry(
        SOURCE_IC_CURRENT,
        GRANULARITY_ENERGY_BINNED,
        session_ids,
        base_dir,
        settings=settings,
    )


def load_sessions_summary(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
    reference_frame: ReferenceFrameKind = REFERENCE_ISO,
) -> dict[str, dict]:
    return _load_sessions_map(
        session_ids,
        lambda sid: load_session_summary_table(
            sid, base_dir, settings=settings, reference_frame=reference_frame,
        ),
    )


def _load_timeslice_position_errors(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
) -> dict | None:
    return _load_registry_columns(
        SOURCE_POSITION_ERROR,
        session_id,
        base_dir,
        LoadOptions(granularity=GRANULARITY_TIMESLICE_SAMPLE, bg_subtract=bg_subtract),
        position_error_to_binned_columns,
    )


def _load_timeslice_sigmas(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
) -> dict | None:
    return _load_registry_columns(
        SOURCE_SIGMA,
        session_id,
        base_dir,
        LoadOptions(granularity=GRANULARITY_TIMESLICE_SAMPLE, bg_subtract=bg_subtract),
        sigma_to_binned_columns,
    )


def _load_timeslice_ic12_diff(
    session_id: str,
    base_dir: str,
    *,
    reference_frame: ReferenceFrameKind,
    bg_subtract: bool = False,
) -> dict | None:
    return _load_registry_columns(
        SOURCE_IC12_POS_DIFF,
        session_id,
        base_dir,
        LoadOptions(
            granularity=GRANULARITY_TIMESLICE_SAMPLE,
            reference_frame=reference_frame,
            bg_subtract=bg_subtract,
        ),
        ic12_to_binned_columns,
    )


def load_session_timeslice_summary_table(
    session_id: str,
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
    reference_frame: ReferenceFrameKind = REFERENCE_ISO,
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

    ic12 = _load_timeslice_ic12_diff(
        session_id, base_dir, reference_frame=reference_frame, bg_subtract=bg_subtract,
    )
    if ic12 is not None:
        if "energy" not in result:
            result.update(ic12)
        else:
            _align_by_length(result, ic12, ("ic12_x_diff", "ic12_y_diff"))

    if "energy" not in result:
        return None
    return result


def load_sessions_timeslice_summary(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
    reference_frame: ReferenceFrameKind = REFERENCE_ISO,
) -> dict[str, dict]:
    return _load_sessions_map(
        session_ids,
        lambda sid: load_session_timeslice_summary_table(
            sid, base_dir, settings=settings, reference_frame=reference_frame,
        ),
    )


def load_sessions_for_source(
    session_ids: Sequence[str],
    base_dir: str,
    source: DataSourceKind,
    *,
    settings: ViewSettings | None = None,
    reference_frame: ReferenceFrameKind = REFERENCE_ISO,
) -> dict[str, dict]:
    if source == DATA_SOURCE_TIMESLICE:
        return load_sessions_timeslice_summary(
            session_ids, base_dir, settings=settings, reference_frame=reference_frame,
        )
    return load_sessions_summary(
        session_ids, base_dir, settings=settings, reference_frame=reference_frame,
    )


def probe_view_option_availability(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    spot_data: dict[str, dict] | None = None,
    registry_availability: dict[str, bool] | None = None,
    settings: ViewSettings | None = None,
) -> dict[str, bool]:
    """Return availability for every unified binned-summary option."""
    from ..data.availability import probe_sessions

    del settings  # registry probes ignore view settings
    spot = spot_data if spot_data is not None else load_sessions_summary(
        session_ids, base_dir,
    )
    spot_y = available_y_groups(spot)
    registry_avail = (
        registry_availability
        if registry_availability is not None
        else probe_sessions(session_ids, base_dir)
    )
    availability: dict[str, bool] = {}
    for opt in VIEW_OPTIONS:
        key = option_key(opt.source, opt.id)
        if opt.id in _REGISTRY_Y_GROUPS:
            availability[key] = registry_avail.get(key, False)
        else:
            availability[key] = opt.id in spot_y
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
