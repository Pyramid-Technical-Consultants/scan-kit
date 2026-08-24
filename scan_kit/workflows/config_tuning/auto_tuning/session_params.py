"""Shared session-ID parsing for auto-tuning workflows."""

from __future__ import annotations

from typing import Any


def parse_session_ids(params: dict[str, Any]) -> list[str]:
    """Return deduplicated session IDs from workflow params."""
    raw = params.get("session_ids")
    if isinstance(raw, list):
        ids = [str(item).strip() for item in raw if str(item).strip()]
    else:
        legacy = str(params.get("session_id", "")).strip()
        ids = [legacy] if legacy else []

    seen: set[str] = set()
    ordered: list[str] = []
    for sid in ids:
        if sid not in seen:
            seen.add(sid)
            ordered.append(sid)
    return ordered
