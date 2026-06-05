"""Suggested PDF filenames for scan-kit reports."""

from __future__ import annotations

import re
from datetime import datetime

from scan_kit.views import ViewEntry

_MAX_STEM_LEN = 96
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


def _session_slug(session_ids: list[str], notes: dict[str, str]) -> str:
    if len(session_ids) == 1:
        sid = session_ids[0]
        note = notes.get(sid, "").strip()
        if note:
            return _slug(note, max_len=28)
        return _slug(sid, max_len=20)

    note_slugs: list[str] = []
    for sid in session_ids:
        note = notes.get(sid, "").strip()
        if not note:
            continue
        slug = _slug(note, max_len=18)
        if slug not in note_slugs:
            note_slugs.append(slug)

    if len(note_slugs) == 1 and len(session_ids) == 2:
        return f"{note_slugs[0]}-2-sessions"
    if len(note_slugs) == 1:
        return f"{note_slugs[0]}-{len(session_ids)}-sessions"
    if note_slugs:
        return f"{note_slugs[0]}-{len(session_ids)}-sessions"

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
    session_part = _session_slug(session_ids, notes)
    views_part = _views_slug(views)

    stem = f"scan-kit-{date_part}_{session_part}_{views_part}"
    if len(stem) <= _MAX_STEM_LEN:
        return f"{stem}.pdf"

    # Drop the longest segment first (usually the views part).
    stem = f"scan-kit-{date_part}_{session_part}_{_views_slug(views[:1])}"
    if len(stem) > _MAX_STEM_LEN:
        stem = f"scan-kit-{date_part}_{session_part}_report"
    if len(stem) > _MAX_STEM_LEN:
        stem = f"scan-kit-{date_part}-report"
    return f"{stem}.pdf"
