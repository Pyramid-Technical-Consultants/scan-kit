"""Registry of data sources: one probe + load per metric."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .cache import (
    clear_cache,
    get_load_cached,
    load_cache_key,
    set_load_cached,
)
from .context import LoadOptions, SessionContext
from .types import DataSourceKind, GranularityKind, data_source_granularity

ProbeFn = Callable[[SessionContext, LoadOptions], bool]
LoadFn = Callable[[SessionContext, LoadOptions], Any]


@dataclass(frozen=True)
class DataSourceSpec:
    id: str
    label: str
    data_sources: frozenset[DataSourceKind]
    supports_bg_subtract: bool
    supports_beam_filter: bool
    probe: ProbeFn
    load: LoadFn
    granularity_for: Mapping[DataSourceKind, GranularityKind] = field(default_factory=dict)

    def granularity_for_data_source(self, data_source: DataSourceKind) -> GranularityKind:
        return self.granularity_for.get(
            data_source,
            data_source_granularity(data_source),
        )


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


def _effective_opts(source_id: str, opts: LoadOptions) -> LoadOptions:
    spec = get_spec(source_id)
    gran = spec.granularity_for_data_source(opts.data_source)
    if gran != opts.granularity:
        return opts.with_granularity(gran)
    return opts


def probe(source_id: str, ctx: SessionContext, opts: LoadOptions) -> bool:
    from .cache import get_probe_cached, probe_cache_key, set_probe_cached

    effective = _effective_opts(source_id, opts)
    key = probe_cache_key(source_id, ctx, effective)
    cached = get_probe_cached(key)
    if cached is not None:
        return cached
    result = get_spec(source_id).probe(ctx, effective)
    set_probe_cached(key, result)
    return result


def load(source_id: str, ctx: SessionContext, opts: LoadOptions) -> Any:
    effective = _effective_opts(source_id, opts)
    key = load_cache_key(source_id, ctx, effective)
    cached = get_load_cached(key)
    if cached is not None:
        return cached
    result = get_spec(source_id).load(ctx, effective)
    set_load_cached(key, result)
    return result
