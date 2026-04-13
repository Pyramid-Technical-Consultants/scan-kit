"""Per-session notes stored as a single JSON file in the data directory."""

from __future__ import annotations

import json
from pathlib import Path

_FILENAME = "session_notes.json"


def load_notes(base_dir: str | Path) -> dict[str, str]:
    """Load all session notes from *base_dir*/session_notes.json.

    Returns an empty dict when the file is missing or unreadable.
    """
    path = Path(base_dir) / _FILENAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def save_note(base_dir: str | Path, session_id: str, text: str) -> None:
    """Write a single session note, merging into the existing file."""
    notes = load_notes(base_dir)
    text = text.strip()
    if text:
        notes[session_id] = text
    else:
        notes.pop(session_id, None)
    path = Path(base_dir) / _FILENAME
    path.write_text(json.dumps(notes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
