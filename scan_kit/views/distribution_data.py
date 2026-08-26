"""Session loaders and cheap availability probes for Distribution Explorer."""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..common.settings import ViewSettings
from .distribution_catalog import (
    MODE_BY_ID,
    MODE_CONFIDENCE_TIMESLICE,
    MODE_GAUSSIAN_FILTER,
    MODE_IC12_POS_DIFF_SPOT,
    MODE_IC12_POS_DIFF_TIMESLICE,
    MODE_POSITION_ERROR_SPOT,
    MODE_POSITION_ERROR_TIMESLICE,
    MODE_POSITION_SPOT,
    MODE_POSITION_TIMESLICE,
    MODE_SIGMA_ERROR_SPOT,
    MODE_SIGMA_ERROR_TIMESLICE,
    MODE_SIGMA_SPOT,
    MODE_SIGMA_TIMESLICE,
    MODES,
    VIEW_OPTIONS,
    resolve_mode_id,
)
from ..data.adapters.distribution import (
    ic12_to_session_xy,
    position_errors_to_session_errors,
    position_to_session_xy,
    sigma_to_session_sigmas,
)
from ..data.context import LoadOptions, SessionContext
from ..data.registry import load as load_source
from ..data.sources.confidence import SOURCE_CONFIDENCE
from ..data.sources.gaussian_fit_filter import SOURCE_GAUSSIAN_FIT_FILTER
from ..data.sources.ic12_pos_diff import SOURCE_IC12_POS_DIFF
from ..data.sources.position import SOURCE_POSITION
from ..data.sources.position_error import SOURCE_POSITION_ERROR
from ..data.sources.sigma import SOURCE_SIGMA
from ..data.sources.sigma_error import SOURCE_SIGMA_ERROR
from ..data.types import (
    COARSE_SOURCE_TIMESLICE,
    DATA_SOURCE_SPOT_ISO,
    DATA_SOURCE_TIMESLICE_ISO,
    coarse_data_source,
)
from .unified_catalog import DataSourceKind, option_key
from ..data.registry import get_spec

_log = logging.getLogger(__name__)

MODE_REGISTRY: dict[str, str] = {
    MODE_POSITION_ERROR_SPOT: SOURCE_POSITION_ERROR,
    MODE_POSITION_ERROR_TIMESLICE: SOURCE_POSITION_ERROR,
    MODE_POSITION_SPOT: SOURCE_POSITION,
    MODE_POSITION_TIMESLICE: SOURCE_POSITION,
    MODE_SIGMA_SPOT: SOURCE_SIGMA,
    MODE_SIGMA_TIMESLICE: SOURCE_SIGMA,
    MODE_SIGMA_ERROR_SPOT: SOURCE_SIGMA_ERROR,
    MODE_SIGMA_ERROR_TIMESLICE: SOURCE_SIGMA_ERROR,
    MODE_IC12_POS_DIFF_SPOT: SOURCE_IC12_POS_DIFF,
    MODE_IC12_POS_DIFF_TIMESLICE: SOURCE_IC12_POS_DIFF,
    MODE_CONFIDENCE_TIMESLICE: SOURCE_CONFIDENCE,
    MODE_GAUSSIAN_FILTER: SOURCE_GAUSSIAN_FIT_FILTER,
}

MODE_ADAPTERS: dict[str, Callable[[Any], Any]] = {
    MODE_POSITION_ERROR_SPOT: position_errors_to_session_errors,
    MODE_POSITION_ERROR_TIMESLICE: position_errors_to_session_errors,
    MODE_POSITION_SPOT: position_to_session_xy,
    MODE_POSITION_TIMESLICE: position_to_session_xy,
    MODE_SIGMA_SPOT: sigma_to_session_sigmas,
    MODE_SIGMA_TIMESLICE: sigma_to_session_sigmas,
    MODE_SIGMA_ERROR_SPOT: position_errors_to_session_errors,
    MODE_SIGMA_ERROR_TIMESLICE: position_errors_to_session_errors,
    MODE_IC12_POS_DIFF_SPOT: ic12_to_session_xy,
    MODE_IC12_POS_DIFF_TIMESLICE: ic12_to_session_xy,
}


def probe_mode_availability(
    session_ids: list[str],
    base_dir: str,
    *,
    source_availability: dict[str, bool] | None = None,
) -> dict[str, bool]:
    """Return which modes have probe-detectable data for the selected sessions."""
    from ..data.availability import probe_sessions

    avail = dict(
        source_availability
        if source_availability is not None
        else probe_sessions(session_ids, base_dir)
    )
    return _extend_mode_availability(avail)


def _extend_mode_availability(avail: dict[str, bool]) -> dict[str, bool]:
    for opt in VIEW_OPTIONS:
        mode_id = resolve_mode_id(opt.id, opt.source)
        if mode_id is not None:
            avail[mode_id] = avail.get(option_key(opt.source, opt.id), False)
    return avail


def probe_session_for_mode(session_id: str, mode: str, base_dir: str) -> bool:
    """Cheap check: session likely has data for *mode* (one layer / headers only)."""
    if mode not in MODE_BY_ID or mode not in MODE_REGISTRY:
        return False
    from ..data.availability import probe_source_option

    source_id = MODE_REGISTRY[mode]
    mode_def = MODE_BY_ID[mode]
    spec = get_spec(source_id)
    for data_source in spec.data_sources:
        if coarse_data_source(data_source) != mode_def.source:
            continue
        if probe_source_option(session_id, base_dir, source_id, data_source):
            return True
    return False


def _load_session_for_mode(
    session_id: str,
    base_dir: str,
    mode: str,
    *,
    settings: ViewSettings | None,
    data_source: DataSourceKind,
) -> Any | None:
    source_id = MODE_REGISTRY[mode]
    payload = load_source(
        source_id,
        SessionContext(session_id, base_dir, settings),
        LoadOptions(
            data_source=data_source,
            bg_subtract=settings.bg_subtract if settings else False,
        ),
    )
    adapter = MODE_ADAPTERS.get(mode)
    if adapter is not None:
        return adapter(payload)
    return payload


def _default_data_source_for_mode(mode: str) -> DataSourceKind:
    mode_def = MODE_BY_ID.get(mode)
    if mode_def is None:
        return DATA_SOURCE_SPOT_ISO
    if mode_def.source == COARSE_SOURCE_TIMESLICE:
        return DATA_SOURCE_TIMESLICE_ISO
    return DATA_SOURCE_SPOT_ISO


def load_sessions_for_mode(
    mode: str,
    session_ids: list[str],
    base_dir: str,
    *,
    settings: ViewSettings | None = None,
    data_source: DataSourceKind | None = None,
) -> dict[str, Any]:
    """Load per-session data for one distribution mode."""
    if mode not in MODE_BY_ID:
        raise ValueError(f"Unknown distribution mode: {mode!r}")
    if mode not in MODE_REGISTRY:
        raise ValueError(f"Mode not registered in data layer: {mode!r}")

    resolved_source = data_source or _default_data_source_for_mode(mode)

    session_data: dict[str, Any] = {}
    for sid in session_ids:
        data = _load_session_for_mode(
            sid, base_dir, mode, settings=settings, data_source=resolved_source,
        )
        if data is not None:
            session_data[sid] = data

    return session_data


def clear_summary_table_cache() -> None:
    """Clear assembled summary-table caches (e.g. after calibration change)."""
    _SUMMARY_TABLE_CACHE.clear()
    _TIMESLICE_SUMMARY_CACHE.clear()


def clear_load_cache() -> None:
    from ..data.cache import clear_cache

    clear_cache()
    clear_summary_table_cache()


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
