"""Session discovery utilities for scan-kit."""

from __future__ import annotations

from pathlib import Path

from .session_meta import SessionMeta
from .session_source import discover_session_entries


def discover_sessions(
    base_dirs: tuple[str, ...] = ("scan_kit", "test_data"),
    project_root: Path | None = None,
) -> list[tuple[str, str, SessionMeta | None]]:
    """Discover sessions (folders, ZIP, tgz, tar.gz, …) and load metadata.

    Args:
        base_dirs: Directories to scan for session data.
        project_root: Root path for the project. Defaults to parent of scan_kit.

    Returns:
        Sorted list of ``(session_id, storage_path, meta)`` for each session.
        *meta* is ``None`` from discovery (fast); load summaries separately if needed.
        Unpacked folders take precedence over an archive with the same id.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    seen: dict[str, tuple[str, SessionMeta | None]] = {}
    for base in base_dirs:
        base_path = Path(base)
        dir_path = base_path if base_path.is_absolute() else project_root / base
        if not dir_path.is_dir():
            continue
        for sid, path_str, meta in discover_session_entries(dir_path):
            if sid in seen:
                continue
            seen[sid] = (path_str, meta)

    return sorted(
        ((sid, info[0], info[1]) for sid, info in seen.items()),
        key=lambda t: t[0],
    )
