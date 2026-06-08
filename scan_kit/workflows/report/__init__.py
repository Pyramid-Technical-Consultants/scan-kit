"""PDF report generation workflow for scan-kit analysis views."""

from __future__ import annotations

from scan_kit.views import VIEW_GROUPS, ViewEntry, view_module_name

REPORT_EXCLUDED_MODULES: frozenset[str] = frozenset({
    "ic_timeslice_replay",
    "ic_timeslice_replay_derived",
    "field_timeslice_replay",
    "session_log_compare",
    "ic_audio_export",
})

GITHUB_URL = "https://github.com/Pyramid-Technical-Consultants/scan-kit"


def reportable_module_names() -> set[str]:
    """Module names that can be included in a PDF report."""
    return {
        view_module_name(entry)
        for _title, entries in VIEW_GROUPS
        for entry in entries
        if view_module_name(entry) not in REPORT_EXCLUDED_MODULES
    }


def report_view_groups() -> list[tuple[str, list[ViewEntry]]]:
    """Return VIEW_GROUPS entries that can be rendered as static matplotlib plots."""
    groups: list[tuple[str, list[ViewEntry]]] = []
    for title, entries in VIEW_GROUPS:
        filtered = [
            entry
            for entry in entries
            if view_module_name(entry) not in REPORT_EXCLUDED_MODULES
        ]
        if filtered:
            groups.append((title, filtered))
    return groups


def default_report_title(
    session_ids: list[str] | None = None,
    notes: dict[str, str] | None = None,
    views: list[ViewEntry] | None = None,
) -> str:
    if session_ids and views is not None:
        from .naming import suggest_report_title

        return suggest_report_title(session_ids, notes or {}, views)
    return "Scan Kit Analysis Report"


def default_report_subtitle(session_ids: list[str]) -> str:
    joined = ", ".join(session_ids)
    prefix = "Sessions: "
    max_len = 96
    if len(prefix) + len(joined) <= max_len:
        return prefix + joined
    budget = max_len - len(prefix) - 3
    return prefix + joined[:budget] + "..."
