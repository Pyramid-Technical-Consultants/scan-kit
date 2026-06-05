"""Default save locations for PDF reports."""

from __future__ import annotations

from pathlib import Path


def _downloads_directory() -> Path:
    try:
        from PySide6.QtCore import QStandardPaths

        location = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DownloadLocation,
        )
        if location:
            path = Path(location)
            if path.is_dir():
                return path
    except Exception:
        pass

    fallback = Path.home() / "Downloads"
    return fallback


def resolve_report_save_dir(last_report_dir: str | None = None) -> Path:
    """Return the last-used report folder, else the system Downloads folder."""
    if last_report_dir:
        path = Path(last_report_dir)
        if path.is_dir():
            return path
        parent = path.parent
        if parent.is_dir():
            return parent
    return _downloads_directory()
