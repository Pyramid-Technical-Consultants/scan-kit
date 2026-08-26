"""Session-level availability probing across registered data sources."""

from __future__ import annotations

from collections.abc import Sequence

from .context import LoadOptions, SessionContext
from .registry import REGISTRY, get_spec, probe as registry_probe
from .types import DataSourceKind, option_key


def probe_source_option(
    session_id: str,
    base_dir: str,
    source_id: str,
    data_source: DataSourceKind,
) -> bool:
    spec = get_spec(source_id)
    if data_source not in spec.data_sources:
        return False
    ctx = SessionContext(session_id, base_dir)
    return registry_probe(
        source_id,
        ctx,
        LoadOptions(data_source=data_source),
    )


def probe_session(
    session_id: str,
    base_dir: str,
    *,
    source_ids: Sequence[str] | None = None,
) -> dict[str, bool]:
    """Return ``option_key(data_source, metric_id)`` availability for one session."""
    ids = source_ids if source_ids is not None else list(REGISTRY.keys())
    availability: dict[str, bool] = {}
    for source_id in ids:
        if source_id not in REGISTRY:
            continue
        spec = REGISTRY[source_id]
        for data_source in sorted(spec.data_sources):
            key = option_key(data_source, source_id)
            availability[key] = probe_source_option(
                session_id,
                base_dir,
                source_id,
                data_source,
            )
    return availability


def probe_sessions(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    source_ids: Sequence[str] | None = None,
) -> dict[str, bool]:
    """Merge availability across sessions (any session with data enables the option)."""
    merged: dict[str, bool] = {}
    for session_id in session_ids:
        session_avail = probe_session(
            session_id,
            base_dir,
            source_ids=source_ids,
        )
        for key, available in session_avail.items():
            merged[key] = merged.get(key, False) or available
    return merged
