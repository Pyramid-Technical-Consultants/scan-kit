"""Registry of data sources: one probe + load per metric."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .cache import (
    clear_cache,
    get_load_cached,
    load_cache_key,
    set_load_cached,
)
from .context import LoadOptions, SessionContext
from .types import GranularityKind, ReferenceFrameKind

ProbeFn = Callable[[SessionContext, LoadOptions], bool]
LoadFn = Callable[[SessionContext, LoadOptions], Any]


@dataclass(frozen=True)
class DataSourceSpec:
    id: str
    label: str
    granularities: frozenset[GranularityKind]
    reference_frames: frozenset[ReferenceFrameKind]
    supports_bg_subtract: bool
    supports_beam_filter: bool
    probe: ProbeFn
    load: LoadFn


REGISTRY: dict[str, DataSourceSpec] = {}


def register(spec: DataSourceSpec) -> DataSourceSpec:
    if spec.id in REGISTRY:
        raise ValueError(f"Duplicate data source id: {spec.id!r}")
    REGISTRY[spec.id] = spec
    return spec


def get_spec(source_id: str) -> DataSourceSpec:
    spec = REGISTRY.get(source_id)
    if spec is None:
        raise KeyError(f"Unknown data source: {source_id!r}")
    return spec


def probe(source_id: str, ctx: SessionContext, opts: LoadOptions) -> bool:
    from .cache import get_probe_cached, probe_cache_key, set_probe_cached

    key = probe_cache_key(source_id, ctx, opts)
    cached = get_probe_cached(key)
    if cached is not None:
        return cached
    result = get_spec(source_id).probe(ctx, opts)
    set_probe_cached(key, result)
    return result


def load(source_id: str, ctx: SessionContext, opts: LoadOptions) -> Any:
    key = load_cache_key(source_id, ctx, opts)
    cached = get_load_cached(key)
    if cached is not None:
        return cached
    result = get_spec(source_id).load(ctx, opts)
    set_load_cached(key, result)
    return result
