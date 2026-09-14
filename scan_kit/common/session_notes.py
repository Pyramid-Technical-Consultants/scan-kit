"""Per-session notes. Legacy JSON is imported once into the machine-local store."""

from __future__ import annotations

import json
from pathlib import Path

_FILENAME = "session_notes.json"


def load_notes_json(base_dir: str | Path) -> dict[str, str]:
    """Read *base_dir*/session_notes.json for one-shot import.

    Returns an empty dict when the file is missing or unreadable.
    """
    path = Path(base_dir) / _FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def load_notes(base_dir: str | Path) -> dict[str, str]:
    """Load notes for *base_dir* from the machine-local store."""
    from .user_store import notes_for_library

    return notes_for_library(base_dir)


def save_note(base_dir: str | Path, session_id: str, text: str) -> None:
    """Write a single session note to the machine-local store."""
    from .user_store import set_session_note

    set_session_note(base_dir, session_id, text)
