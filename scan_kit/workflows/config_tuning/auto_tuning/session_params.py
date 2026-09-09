"""Shared session-ID parsing and validation for auto-tuning workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scan_kit.common.session_source import resolve_session_source


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


def validate_session_selection(params: dict[str, Any]) -> list[str]:
    """Validate the session browser inputs every session-driven workflow shares."""
    errors: list[str] = []
    session_ids = parse_session_ids(params)
    if not session_ids:
        errors.append("Select at least one session.")
    data_dir = str(params.get("data_dir", "")).strip()
    if not data_dir:
        errors.append("Enter the folder containing session data.")
    elif not Path(data_dir).expanduser().is_dir():
        errors.append("Session data folder is not a directory.")
    else:
        base = Path(data_dir).expanduser().resolve()
        missing = [
            sid for sid in session_ids if resolve_session_source(sid, base) is None
        ]
        if missing and len(missing) == len(session_ids):
            errors.append(f"No selected sessions were found under {base}.")
    return errors
