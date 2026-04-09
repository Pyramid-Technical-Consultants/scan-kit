"""Session discovery utilities for scan-kit."""

from __future__ import annotations

from pathlib import Path

from .io import SessionMeta, load_termination_summary


def discover_sessions(
    base_dirs: tuple[str, ...] = ("scan_kit", "test_data"),
    project_root: Path | None = None,
) -> list[tuple[str, str, SessionMeta | None]]:
    """Discover session ZIPs and load their metadata.

    Args:
        base_dirs: Directories to scan for {session_id}.zip files.
        project_root: Root path for the project. Defaults to parent of scan_kit.

    Returns:
        Sorted list of ``(session_id, zip_path, meta)`` tuples, where *meta*
        is a :class:`SessionMeta` parsed from ``termination_summary.txt``
        (or ``None`` if the file is missing / unparseable).
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    seen: dict[str, tuple[str, SessionMeta | None]] = {}
    for base in base_dirs:
        base_path = Path(base)
        dir_path = base_path if base_path.is_absolute() else project_root / base
        if not dir_path.is_dir():
            continue
        for zp in dir_path.glob("*.zip"):
            sid = zp.stem
            if sid in seen:
                continue
            meta = load_termination_summary(str(zp), sid)
            seen[sid] = (str(zp), meta)

    return sorted(
        ((sid, info[0], info[1]) for sid, info in seen.items()),
        key=lambda t: t[0],
    )
