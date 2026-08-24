"""Session loaders and cheap availability probes for Distribution Explorer."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..common import (
    C_X_POSITION,
    C_Y_POSITION,
    create_valid_mask,
    detect_beam_on_mask,
    load_session_raw,
    process_position_data,
    try_load_position_data,
)
from ..common.session_ic_xy import SessionIcXYData
from ..common.timeslice_sigma import (
    TIMESLICE_SIGMA_COLS,
    TIMESLICE_SIGMA_ERROR_COLS,
    SessionIcSigmas,
    frame_timeslice_sigma_arrays,
    frame_timeslice_sigma_error_arrays,
    load_session_beam_on_sigma_errors,
    load_session_beam_on_sigmas,
    resolve_timeslice_sigma_source,
    timeslice_sigma_error_available,
)
from ..common.schema import (
    C_IC1_X_POS,
    C_IC1_Y_POS,
    C_IC2_X_POS,
    C_IC2_Y_POS,
    POSITION_KEY_G2,
    POSITION_KEY_G3,
    resolve_concept_column,
)
from ..common.session_source import (
    load_session_timeslice_device_units,
    read_session_csv_columns,
    resolve_session_source,
)
from ..common.settings import ViewSettings
from ..common.timeslice_confidence import (
    TIMESLICE_CONFIDENCE_COLS,
    load_session_beam_on_confidence_correlations,
    resolve_timeslice_confidence_source,
)
from ..common.timeslice_gaussian_fit_filter_coverage import (
    TIMESLICE_GAUSSIAN_FIT_FILTER_COVERAGE_COLS,
    compute_session_gaussian_fit_filter_coverage,
    resolve_timeslice_gaussian_fit_filter_coverage_source,
)
from ..common.timeslice_position_error import (
    TIMESLICE_POSITION_ERROR_COLS,
    SessionPositionErrors,
    frame_timeslice_error_arrays,
    frame_timeslice_iso_position_arrays,
    load_session_beam_on_iso_positions,
    load_session_beam_on_position_errors,
    resolve_session_timeslice_error_source,
    resolve_session_timeslice_iso_position_source,
)
from .distribution_catalog import (
    MODE_BY_ID,
    MODE_CONFIDENCE_TIMESLICE,
    MODE_GAUSSIAN_FILTER,
    MODE_POSITION_ERROR_SPOT,
    MODE_POSITION_ERROR_TIMESLICE,
    MODE_POSITION_SPOT,
    MODE_POSITION_TIMESLICE,
    MODE_SIGMA_ERROR_TIMESLICE,
    MODE_SIGMA_SPOT,
    MODE_SIGMA_TIMESLICE,
    MODES,
    VIEW_OPTIONS,
    resolve_mode_id,
)
from .unified_catalog import option_key

_log = logging.getLogger(__name__)

_LOAD_CACHE: dict[tuple, dict[str, Any]] = {}


def _cache_key(
    mode: str,
    session_ids: list[str],
    base_dir: str,
    settings: ViewSettings | None,
) -> tuple:
    bg = settings.bg_subtract if settings else False
    return (mode, tuple(session_ids), base_dir, bg)


_SIG_KEY_VARIANTS = ("spot_sigma_raw", "spot_sigma")
_SPOT_SIGMA_ATTRS = (
    ("ic1_x", "ic1", "x"),
    ("ic1_y", "ic1", "y"),
    ("ic2_x", "ic2", "x"),
    ("ic2_y", "ic2", "y"),
)


def _resolve_sigma_col(columns, ic: str, axis: str) -> str | None:
    for key in _SIG_KEY_VARIANTS:
        for prefix in (f"r_{ic}_{axis}_{key}", f"{ic}_{axis}_{key}"):
            if prefix in columns:
                return prefix
    return None


def _probe_spot_sigmas(src) -> bool:
    spot_cols = read_session_csv_columns(src, "spot_data.csv")
    if not spot_cols:
        return False
    return all(
        _resolve_sigma_col(spot_cols, ic, axis) is not None
        for _attr, ic, axis in _SPOT_SIGMA_ATTRS
    )


def _load_spot_sigmas(session_id: str, base_dir: str) -> SessionIcSigmas | None:
    _input_map, spot_data = load_session_raw(session_id, base_dir=base_dir)
    if spot_data is None:
        return None

    cols: dict[str, str] = {}
    for attr, ic, axis in _SPOT_SIGMA_ATTRS:
        col = _resolve_sigma_col(spot_data.columns, ic, axis)
        if col is None:
            return None
        cols[attr] = col

    frame = spot_data[list(cols.values())].apply(pd.to_numeric, errors="coerce")
    clean = frame[create_valid_mask(frame)]
    if clean.empty:
        return None

    return SessionIcSigmas(
        ic1_x=clean[cols["ic1_x"]].to_numpy(dtype=float) * 2.0,
        ic1_y=clean[cols["ic1_y"]].to_numpy(dtype=float) * 2.0,
        ic2_x=clean[cols["ic2_x"]].to_numpy(dtype=float) * 2.0,
        ic2_y=clean[cols["ic2_y"]].to_numpy(dtype=float) * 2.0,
    )


def _process_spot_position_errors(session_id: str, position_key: str, base_dir: str):
    data = process_position_data(
        session_id,
        position_key,
        extra_input_columns=[C_X_POSITION, C_Y_POSITION],
        base_dir=base_dir,
    )
    if data is None:
        return None
    if C_X_POSITION not in data or C_Y_POSITION not in data:
        _log.debug(
            "Session %s: input_map missing plan position columns; skipping",
            session_id,
        )
        return None

    plan_x = np.asarray(data[C_X_POSITION], dtype=float)
    plan_y = np.asarray(data[C_Y_POSITION], dtype=float)
    return SessionPositionErrors(
        ic1_x=np.asarray(data["ic1_x"], dtype=float) - plan_x,
        ic1_y=np.asarray(data["ic1_y"], dtype=float) - plan_y,
        ic2_x=np.asarray(data["ic2_x"], dtype=float) - plan_x,
        ic2_y=np.asarray(data["ic2_y"], dtype=float) - plan_y,
    )


def _process_spot_positions(session_id: str, position_key: str, base_dir: str):
    data = process_position_data(
        session_id,
        position_key,
        extra_input_columns=[C_X_POSITION, C_Y_POSITION],
        base_dir=base_dir,
    )
    if data is None:
        return None
    if C_X_POSITION not in data or C_Y_POSITION not in data:
        _log.debug(
            "Session %s: input_map missing plan position columns; skipping",
            session_id,
        )
        return None

    return SessionIcXYData(
        plan_x=np.asarray(data[C_X_POSITION], dtype=float),
        plan_y=np.asarray(data[C_Y_POSITION], dtype=float),
        ic1_x=np.asarray(data["ic1_x"], dtype=float),
        ic1_y=np.asarray(data["ic1_y"], dtype=float),
        ic2_x=np.asarray(data["ic2_x"], dtype=float),
        ic2_y=np.asarray(data["ic2_y"], dtype=float),
    )


def _spot_columns_available(spot_cols: list[str], position_key: str) -> bool:
    for concept in (C_IC1_X_POS, C_IC1_Y_POS, C_IC2_X_POS, C_IC2_Y_POS):
        if resolve_concept_column(spot_cols, concept, position_key=position_key) is None:
            return False
    return True


def _probe_spot_position_errors(src) -> bool:
    input_cols = read_session_csv_columns(src, "input_map.csv")
    spot_cols = read_session_csv_columns(src, "spot_data.csv")
    if not input_cols or not spot_cols:
        return False
    if resolve_concept_column(input_cols, C_X_POSITION) is None:
        return False
    if resolve_concept_column(input_cols, C_Y_POSITION) is None:
        return False
    return any(
        _spot_columns_available(spot_cols, position_key)
        for position_key in (POSITION_KEY_G3, POSITION_KEY_G2)
    )


def _probe_timeslice_frame(
    src,
    *,
    usecols: list[str],
    resolve_source: Callable,
    frame_arrays: Callable,
) -> bool:
    frames = load_session_timeslice_device_units(
        src, usecols=usecols, max_frames=1,
    )
    if not frames:
        return False
    source = resolve_source(frames[0].columns)
    if source is None:
        return False
    beam_on = detect_beam_on_mask(frames[0])
    if beam_on is None or not np.any(beam_on):
        return False
    return frame_arrays(frames[0], source) is not None


def _probe_spot_positions(src) -> bool:
    input_cols = read_session_csv_columns(src, "input_map.csv")
    spot_cols = read_session_csv_columns(src, "spot_data.csv")
    if not input_cols or not spot_cols:
        return False
    if resolve_concept_column(input_cols, C_X_POSITION) is None:
        return False
    if resolve_concept_column(input_cols, C_Y_POSITION) is None:
        return False
    return any(
        _spot_columns_available(spot_cols, position_key)
        for position_key in (POSITION_KEY_G3, POSITION_KEY_G2)
    )


def _probe_position_timeslice(src) -> bool:
    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS, max_frames=1,
    )
    if not frames:
        return False
    iso_source = resolve_session_timeslice_iso_position_source(src, frames)
    if iso_source is None:
        return False
    beam_on = detect_beam_on_mask(frames[0])
    if beam_on is None or not np.any(beam_on):
        return False
    return frame_timeslice_iso_position_arrays(frames[0], iso_source) is not None


def _probe_sigma_error_timeslice(src) -> bool:
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
    from ..common.timeslice_sigma import _resolve_sigma_target_columns

    target_cols = _resolve_sigma_target_columns(frames[0].columns)
    if target_cols is None:
        return False
    return (
        frame_timeslice_sigma_error_arrays(frames[0], source, target_cols)
        is not None
    )


def _probe_position_error_timeslice(src) -> bool:
    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS, max_frames=1,
    )
    if not frames:
        return False
    error_source = resolve_session_timeslice_error_source(src, frames)
    if error_source is None:
        return False
    beam_on = detect_beam_on_mask(frames[0])
    if beam_on is None or not np.any(beam_on):
        return False
    return frame_timeslice_error_arrays(frames[0], error_source) is not None


def probe_session_for_mode(session_id: str, mode: str, base_dir: str) -> bool:
    """Cheap check: session likely has data for *mode* (one layer / headers only)."""
    if mode not in MODE_BY_ID:
        return False

    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return False

    if mode == MODE_POSITION_ERROR_SPOT:
        return _probe_spot_position_errors(src)

    if mode == MODE_POSITION_SPOT:
        return _probe_spot_positions(src)

    if mode == MODE_SIGMA_SPOT:
        return _probe_spot_sigmas(src)

    if mode == MODE_POSITION_ERROR_TIMESLICE:
        return _probe_position_error_timeslice(src)

    if mode == MODE_POSITION_TIMESLICE:
        return _probe_position_timeslice(src)

    if mode == MODE_SIGMA_TIMESLICE:
        return _probe_timeslice_frame(
            src,
            usecols=TIMESLICE_SIGMA_COLS,
            resolve_source=resolve_timeslice_sigma_source,
            frame_arrays=frame_timeslice_sigma_arrays,
        )

    if mode == MODE_SIGMA_ERROR_TIMESLICE:
        return _probe_sigma_error_timeslice(src)

    if mode == MODE_CONFIDENCE_TIMESLICE:
        frames = load_session_timeslice_device_units(
            src, usecols=TIMESLICE_CONFIDENCE_COLS, max_frames=1,
        )
        if not frames:
            return False
        if resolve_timeslice_confidence_source(frames[0].columns) is None:
            return False
        beam_on = detect_beam_on_mask(frames[0])
        return beam_on is not None and np.any(beam_on)

    if mode == MODE_GAUSSIAN_FILTER:
        frames = load_session_timeslice_device_units(
            src,
            usecols=TIMESLICE_GAUSSIAN_FIT_FILTER_COVERAGE_COLS,
            max_frames=1,
        )
        if not frames:
            return False
        return (
            resolve_timeslice_gaussian_fit_filter_coverage_source(frames[0].columns)
            is not None
        )

    return False


def probe_mode_availability(
    session_ids: list[str],
    base_dir: str,
) -> dict[str, bool]:
    """Return which modes have probe-detectable data for the selected sessions."""
    avail = {
        mode: any(
            probe_session_for_mode(sid, mode, base_dir) for sid in session_ids
        )
        for mode in MODE_BY_ID
    }
    for opt in VIEW_OPTIONS:
        mode_id = resolve_mode_id(opt.id, opt.source)
        if mode_id is not None:
            avail[option_key(opt.source, opt.id)] = avail.get(mode_id, False)
    return avail


def load_sessions_for_mode(
    mode: str,
    session_ids: list[str],
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
) -> dict[str, Any]:
    """Load per-session data for one distribution mode."""
    if mode not in MODE_BY_ID:
        raise ValueError(f"Unknown distribution mode: {mode!r}")

    key = _cache_key(mode, session_ids, base_dir, settings)
    cached = _LOAD_CACHE.get(key)
    if cached is not None:
        return cached

    bg_subtract = settings.bg_subtract if settings else False
    session_data: dict[str, Any] = {}

    if mode == MODE_POSITION_ERROR_TIMESLICE:
        for sid in session_ids:
            errors = load_session_beam_on_position_errors(
                sid, base_dir, bg_subtract=bg_subtract,
            )
            if errors is not None:
                session_data[sid] = errors

    elif mode == MODE_POSITION_ERROR_SPOT:
        for sid in session_ids:
            errors = try_load_position_data(
                sid, base_dir, _process_spot_position_errors, raw=False,
            )
            if errors is not None:
                session_data[sid] = errors

    elif mode == MODE_POSITION_SPOT:
        for sid in session_ids:
            positions = try_load_position_data(
                sid, base_dir, _process_spot_positions, raw=False,
            )
            if positions is not None:
                session_data[sid] = positions

    elif mode == MODE_SIGMA_SPOT:
        for sid in session_ids:
            sigmas = _load_spot_sigmas(sid, base_dir)
            if sigmas is not None:
                session_data[sid] = sigmas

    elif mode == MODE_SIGMA_TIMESLICE:
        for sid in session_ids:
            sigmas = load_session_beam_on_sigmas(sid, base_dir, bg_subtract=bg_subtract)
            if sigmas is not None:
                session_data[sid] = sigmas

    elif mode == MODE_POSITION_TIMESLICE:
        for sid in session_ids:
            positions = load_session_beam_on_iso_positions(
                sid, base_dir, bg_subtract=bg_subtract,
            )
            if positions is not None:
                session_data[sid] = positions

    elif mode == MODE_SIGMA_ERROR_TIMESLICE:
        for sid in session_ids:
            errors = load_session_beam_on_sigma_errors(
                sid, base_dir, bg_subtract=bg_subtract,
            )
            if errors is not None:
                session_data[sid] = errors

    elif mode == MODE_CONFIDENCE_TIMESLICE:
        for sid in session_ids:
            samples = load_session_beam_on_confidence_correlations(
                sid, base_dir, bg_subtract=bg_subtract,
            )
            if samples is not None:
                session_data[sid] = samples

    elif mode == MODE_GAUSSIAN_FILTER:
        for sid in session_ids:
            coverage = compute_session_gaussian_fit_filter_coverage(
                sid, base_dir, bg_subtract=bg_subtract,
            )
            if coverage is not None:
                session_data[sid] = coverage

    _LOAD_CACHE[key] = session_data
    return session_data


def clear_load_cache() -> None:
    _LOAD_CACHE.clear()


def mode_has_data(
    mode: str,
    session_ids: list[str],
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
    availability: dict[str, bool] | None = None,
) -> bool:
    del settings  # probes ignore bg_subtract; full load uses cache when needed
    if availability is not None:
        return availability.get(mode, False)
    return probe_mode_availability(session_ids, base_dir).get(mode, False)


def default_mode(
    session_ids: list[str],
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
    availability: dict[str, bool] | None = None,
) -> str:
    del settings
    avail = availability or probe_mode_availability(session_ids, base_dir)
    for mode_def in MODES:
        if avail.get(mode_def.id, False):
            return mode_def.id
    return MODE_POSITION_ERROR_SPOT
