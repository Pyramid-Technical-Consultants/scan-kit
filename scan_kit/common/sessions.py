"""Session discovery utilities for scan-kit."""

from pathlib import Path


def discover_sessions(
    base_dirs: tuple[str, ...] = ("scan_kit", "test_data"),
    project_root: Path | None = None,
) -> list[str]:
    """Return sorted list of session IDs from ZIP files in base dirs.

    Args:
        base_dirs: Directories to scan for {session_id}.zip files.
        project_root: Root path for the project. Defaults to parent of scan_kit.

    Returns:
        Sorted, deduplicated list of session ID strings.
    """
    if project_root is None:
        # Assume we're in scan_kit/common/, so project root is 2 levels up
        project_root = Path(__file__).resolve().parent.parent.parent

    seen: set[str] = set()
    for base in base_dirs:
        base_path = Path(base)
        dir_path = base_path if base_path.is_absolute() else project_root / base
        if not dir_path.is_dir():
            continue
        for zip_path in dir_path.glob("*.zip"):
            session_id = zip_path.stem
            seen.add(session_id)

    return sorted(seen)
