"""Analysis view modules for scan-kit.

Launcher metadata avoids importing view modules (heavy matplotlib/pandas stack)
until a view is actually run. Use :data:`VIEW_GROUPS` / :data:`VIEWS` for
(display name, module name, description) tuples.

Optional **Audio Explorer** is detected via :func:`importlib.util.find_spec`
without importing :mod:`sounddevice`.
"""

from __future__ import annotations

ViewEntry = tuple[str, str, str]


def view_module_name(entry: ViewEntry) -> str:
    return entry[1]


def view_description(entry: ViewEntry) -> str:
    return entry[2]


# Analysis views run in the Rust shell. These lists stay empty so the Qt launcher
# no longer imports the deleted windows.
_UNIFIED_VIEWS: list[ViewEntry] = []
_SPECIALIZED_VIEWS: list[ViewEntry] = []

VIEW_GROUPS: list[tuple[str, list[ViewEntry]]] = [
    ("Unified Views", _UNIFIED_VIEWS),
    ("Specialized Analysis", _SPECIALIZED_VIEWS),
]

VIEWS: list[ViewEntry] = [entry for _title, entries in VIEW_GROUPS for entry in entries]

# Tkinter views cannot share a process with the Qt warm-worker (matplotlib backend clash).
TK_ONLY_VIEW_MODULES: frozenset[str] = frozenset()
