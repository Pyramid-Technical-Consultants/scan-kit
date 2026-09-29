"""Session discovery utilities for scan-kit."""

from __future__ import annotations

from pathlib import Path

from .data_location import is_remote_location, local_path
from .session_meta import SessionMeta
from .session_source import discover_session_entries


def discover_sessions(
    base_dirs: tuple[str, ...] = ("scan_kit", "test_data"),
    project_root: Path | None = None,
) -> list[tuple[str, str, SessionMeta | None]]:
    """Discover sessions (folders, ZIP, tgz, tar.gz, …) and load metadata.

    Args:
        base_dirs: Directories or fsspec URLs to scan for session data.
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
        if is_remote_location(base):
            entries = discover_session_entries(base)
        else:
            dir_path = local_path(base)
            if dir_path is None:
                continue
            if not dir_path.is_absolute():
                dir_path = project_root / dir_path
            if not dir_path.is_dir():
                continue
            entries = discover_session_entries(dir_path)
        for sid, path_str, meta in entries:
            if sid in seen:
                continue
            seen[sid] = (path_str, meta)

    return sorted(
        ((sid, info[0], info[1]) for sid, info in seen.items()),
        key=lambda t: t[0],
    )
