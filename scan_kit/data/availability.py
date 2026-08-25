"""Session-level availability probing across registered data sources."""

from __future__ import annotations

from collections.abc import Sequence

from .cache import get_probe_cached, probe_cache_key, set_probe_cached
from .context import LoadOptions, SessionContext
from .registry import REGISTRY, get_spec, probe as registry_probe
from .sources.ic12_pos_diff import SOURCE_IC12_POS_DIFF
from .types import (
    DATA_SOURCE_SPOT,
    DATA_SOURCE_TIMESLICE,
    GRANULARITY_ENERGY_BINNED,
    GRANULARITY_LAYER,
    GRANULARITY_SESSION_COMPUTE,
    GRANULARITY_SPOT,
    GRANULARITY_TIMESLICE_SAMPLE,
    REFERENCE_CHAMBER,
    REFERENCE_ISO,
    DataSourceKind,
    GranularityKind,
    ReferenceFrameKind,
    option_key,
)

_DEFAULT_REFERENCE_FRAMES: tuple[ReferenceFrameKind, ...] = (
    REFERENCE_ISO,
    REFERENCE_CHAMBER,
)


def granularity_to_data_source(granularity: GranularityKind) -> DataSourceKind:
    if granularity == GRANULARITY_SPOT:
        return DATA_SOURCE_SPOT
    if granularity in (
        GRANULARITY_TIMESLICE_SAMPLE,
        GRANULARITY_SESSION_COMPUTE,
        GRANULARITY_ENERGY_BINNED,
    ):
        return DATA_SOURCE_TIMESLICE
    if granularity == GRANULARITY_LAYER:
        return DATA_SOURCE_SPOT
    return DATA_SOURCE_SPOT


def _cached_probe(
    source_id: str,
    ctx: SessionContext,
    opts: LoadOptions,
) -> bool:
    key = probe_cache_key(source_id, ctx, opts)
    cached = get_probe_cached(key)
    if cached is not None:
        return cached
    result = registry_probe(source_id, ctx, opts)
    set_probe_cached(key, result)
    return result


def _reference_frames_for_probe(
    spec_reference_frames: frozenset[ReferenceFrameKind],
    *,
    reference_frames: Sequence[ReferenceFrameKind],
) -> tuple[ReferenceFrameKind, ...]:
    if not spec_reference_frames:
        return (REFERENCE_ISO,)
    return tuple(
        rf for rf in reference_frames if rf in spec_reference_frames
    )


def probe_source_option(
    session_id: str,
    base_dir: str,
    source_id: str,
    granularity: GranularityKind,
    *,
    reference_frames: Sequence[ReferenceFrameKind] = _DEFAULT_REFERENCE_FRAMES,
) -> bool:
    spec = get_spec(source_id)
    ctx = SessionContext(session_id, base_dir)
    frames = _reference_frames_for_probe(
        spec.reference_frames,
        reference_frames=reference_frames,
    )

    if source_id == SOURCE_IC12_POS_DIFF and granularity == GRANULARITY_TIMESLICE_SAMPLE:
        return (
            _cached_probe(
                source_id, ctx,
                LoadOptions(granularity=granularity, reference_frame=REFERENCE_ISO),
            )
            or _cached_probe(
                source_id, ctx,
                LoadOptions(granularity=granularity, reference_frame=REFERENCE_CHAMBER),
            )
        )

    if granularity == GRANULARITY_SPOT and len(frames) > 1:
        return any(
            _cached_probe(
                source_id, ctx,
                LoadOptions(granularity=granularity, reference_frame=rf),
            )
            for rf in frames
        )

    if granularity == GRANULARITY_TIMESLICE_SAMPLE and spec.reference_frames:
        return any(
            _cached_probe(
                source_id, ctx,
                LoadOptions(granularity=granularity, reference_frame=rf),
            )
            for rf in frames
        )

    return _cached_probe(
        source_id,
        ctx,
        LoadOptions(granularity=granularity, reference_frame=frames[0]),
    )


def probe_session(
    session_id: str,
    base_dir: str,
    *,
    source_ids: Sequence[str] | None = None,
    reference_frames: Sequence[ReferenceFrameKind] = _DEFAULT_REFERENCE_FRAMES,
) -> dict[str, bool]:
    """Return ``option_key(source, metric_id)`` availability for one session."""
    ids = source_ids if source_ids is not None else list(REGISTRY.keys())
    availability: dict[str, bool] = {}
    for source_id in ids:
        if source_id not in REGISTRY:
            continue
        spec = REGISTRY[source_id]
        for granularity in spec.granularities:
            source_kind = granularity_to_data_source(granularity)
            key = option_key(source_kind, source_id)
            availability[key] = probe_source_option(
                session_id,
                base_dir,
                source_id,
                granularity,
                reference_frames=reference_frames,
            )
    return availability


def probe_sessions(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    source_ids: Sequence[str] | None = None,
    reference_frames: Sequence[ReferenceFrameKind] = _DEFAULT_REFERENCE_FRAMES,
) -> dict[str, bool]:
    """Merge availability across sessions (any session with data enables the option)."""
    merged: dict[str, bool] = {}
    for session_id in session_ids:
        session_avail = probe_session(
            session_id,
            base_dir,
            source_ids=source_ids,
            reference_frames=reference_frames,
        )
        for key, available in session_avail.items():
            merged[key] = merged.get(key, False) or available
    return merged
