"""Unified probe/load caches for the data-source registry."""

from __future__ import annotations

from typing import Any

from .context import LoadOptions, SessionContext

_PROBE_CACHE: dict[tuple, bool] = {}
_LOAD_CACHE: dict[tuple, Any] = {}


def probe_cache_key(
    source_id: str,
    ctx: SessionContext,
    opts: LoadOptions,
) -> tuple:
    return (
        "probe",
        source_id,
        ctx.session_id,
        ctx.base_dir,
        opts.granularity,
        opts.reference_frame,
    )


def load_cache_key(
    source_id: str,
    ctx: SessionContext,
    opts: LoadOptions,
) -> tuple:
    return (
        "load",
        source_id,
        ctx.session_id,
        ctx.base_dir,
        opts.granularity,
        opts.reference_frame,
        opts.resolved_bg_subtract(ctx),
    )


def get_probe_cached(key: tuple) -> bool | None:
    return _PROBE_CACHE.get(key)


def set_probe_cached(key: tuple, value: bool) -> None:
    _PROBE_CACHE[key] = value


def get_load_cached(key: tuple) -> Any | None:
    if key in _LOAD_CACHE:
        return _LOAD_CACHE[key]
    return None


def set_load_cached(key: tuple, value: Any) -> None:
    _LOAD_CACHE[key] = value


def clear_cache() -> None:
    _PROBE_CACHE.clear()
    _LOAD_CACHE.clear()
