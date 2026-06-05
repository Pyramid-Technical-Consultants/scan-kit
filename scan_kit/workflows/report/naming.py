"""Suggested PDF filenames for scan-kit reports."""

from __future__ import annotations

import re
from datetime import datetime

from scan_kit.views import ViewEntry

_MAX_STEM_LEN = 96
_MAX_TITLE_LEN = 120
_SLUG_RE = re.compile(r"[^\w\s-]+", re.UNICODE)


def _slug(text: str, *, max_len: int = 32) -> str:
    cleaned = _SLUG_RE.sub("", text.lower().strip())
    cleaned = re.sub(r"[\s_]+", "-", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    if not cleaned:
        return "report"
    if len(cleaned) <= max_len:
        return cleaned
    trimmed = cleaned[:max_len].rstrip("-")
    return trimmed or "report"


def _module_slug(module_name: str, *, max_len: int = 24) -> str:
    return _slug(module_name.replace("_", "-"), max_len=max_len)


def _unique_session_notes(session_ids: list[str], notes: dict[str, str]) -> list[str]:
    seen: list[str] = []
    for sid in session_ids:
        note = notes.get(sid, "").strip()
        if note and note not in seen:
            seen.append(note)
    return seen


def _session_notes_summary(session_ids: list[str], notes: dict[str, str]) -> str:
    """Readable combined summary of session notes (falls back to session ids/count)."""
    unique_notes = _unique_session_notes(session_ids, notes)
    if len(unique_notes) == 1:
        note = unique_notes[0]
        if len(session_ids) > 1:
            return f"{note} ({len(session_ids)} sessions)"
        return note
    if len(unique_notes) == 2:
        return f"{unique_notes[0]} and {unique_notes[1]}"
    if len(unique_notes) > 2:
        head = "; ".join(unique_notes[:-1])
        return f"{head}; and {unique_notes[-1]}"

    if len(session_ids) == 1:
        return session_ids[0]
    if len(session_ids) == 2:
        return f"{session_ids[0]} and 1 other session"
    return f"{len(session_ids)} sessions"


def _views_label(views: list[ViewEntry]) -> str:
    if not views:
        return "Analysis Report"
    if len(views) == 1:
        return views[0][0]
    if len(views) == 2:
        return f"{views[0][0]} and {views[1][0]}"
    return f"{len(views)} analysis views"


def _truncate_text(text: str, *, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def suggest_report_title(
    session_ids: list[str],
    notes: dict[str, str],
    views: list[ViewEntry],
) -> str:
    """Build a readable default report title from selected views and session notes."""
    views_part = _views_label(views)
    session_part = _session_notes_summary(session_ids, notes)
    return _truncate_text(f"{views_part} — {session_part}", max_len=_MAX_TITLE_LEN)


def _session_notes_slug(session_ids: list[str], notes: dict[str, str]) -> str:
    unique_notes = _unique_session_notes(session_ids, notes)
    if len(unique_notes) == 1:
        note_slug = _slug(unique_notes[0], max_len=28)
        if len(session_ids) > 1:
            return f"{note_slug}-{len(session_ids)}-sessions"
        return note_slug
    if len(unique_notes) == 2:
        left = _slug(unique_notes[0], max_len=16)
        right = _slug(unique_notes[1], max_len=16)
        combined = f"{left}-and-{right}"
        return combined if len(combined) <= 40 else f"{len(unique_notes)}-notes"
    if len(unique_notes) > 2:
        return f"{len(unique_notes)}-notes"

    if len(session_ids) == 1:
        return _slug(session_ids[0], max_len=20)
    if len(session_ids) == 2:
        return f"{_slug(session_ids[0], max_len=12)}-and-1-more"
    return f"{len(session_ids)}-sessions"


def _views_slug(views: list[ViewEntry]) -> str:
    if not views:
        return "report"
    if len(views) == 1:
        return _module_slug(views[0][1])
    if len(views) == 2:
        left = _module_slug(views[0][1], max_len=18)
        right = _module_slug(views[1][1], max_len=18)
        combined = f"{left}-and-{right}"
        return combined if len(combined) <= 44 else "2-views"
    return f"{len(views)}-views"


def suggest_report_filename(
    session_ids: list[str],
    notes: dict[str, str],
    views: list[ViewEntry],
    *,
    generated_at: datetime | None = None,
) -> str:
    """Build a readable, filesystem-safe default report filename."""
    when = generated_at or datetime.now()
    date_part = when.strftime("%Y-%m-%d")
    views_part = _views_slug(views)
    session_part = _session_notes_slug(session_ids, notes)

    stem = f"{date_part}_{views_part}_{session_part}"
    if len(stem) <= _MAX_STEM_LEN:
        return f"{stem}.pdf"

    # Shorten the views segment first, then drop the session summary.
    stem = f"{date_part}_{_views_slug(views[:1])}_{session_part}"
    if len(stem) > _MAX_STEM_LEN:
        stem = f"{date_part}_{_views_slug(views[:1])}_report"
    if len(stem) > _MAX_STEM_LEN:
        stem = f"{date_part}-report"
    return f"{stem}.pdf"
