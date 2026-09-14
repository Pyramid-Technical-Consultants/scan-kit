"""Send files and folders to the OS recycle bin / trash."""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def move_to_trash(path: str | Path) -> None:
    """Move *path* to the Recycle Bin (Windows) or trash (POSIX).

    Missing paths are ignored. Raises ``OSError`` if the OS refuse the move.
    """
    target = Path(path)
    if not target.exists():
        return
    if sys.platform == "win32":
        _windows_recycle(target)
    elif sys.platform == "darwin":
        _darwin_recycle(target)
    else:
        _posix_recycle(target)


def _windows_recycle(path: Path) -> None:
    import ctypes
    from ctypes import wintypes

    FO_DELETE = 3
    FOF_SILENT = 0x0004
    FOF_NOCONFIRMATION = 0x0010
    FOF_ALLOWUNDO = 0x0040
    FOF_NOERRORUI = 0x0400

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR),
            ("fFlags", wintypes.WORD),
            ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", wintypes.LPVOID),
            ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    # SHFileOperationW wants a double-null-terminated writable buffer.
    buf = ctypes.create_unicode_buffer(str(path.resolve()) + "\0")
    op = SHFILEOPSTRUCTW()
    op.hwnd = None
    op.wFunc = FO_DELETE
    op.pFrom = ctypes.cast(buf, wintypes.LPCWSTR)
    op.pTo = None
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_SILENT
    rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if rc:
        raise OSError(rc, f"Could not send {path} to the Recycle Bin")


def _darwin_recycle(path: Path) -> None:
    trash = Path.home() / ".Trash"
    trash.mkdir(parents=True, exist_ok=True)
    dest = _unique_name(trash, path.name)
    shutil.move(str(path), dest)


def _posix_recycle(path: Path) -> None:
    gio = shutil.which("gio")
    if gio is not None:
        subprocess.run([gio, "trash", str(path)], check=True)
        return
    trash_home = Path.home() / ".local/share/Trash"
    files_dir = trash_home / "files"
    info_dir = trash_home / "info"
    files_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_name(files_dir, path.name)
    shutil.move(str(path), dest)
    deletion_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    info = (
        "[Trash Info]\n"
        f"Path={path.resolve().as_posix()}\n"
        f"DeletionDate={deletion_date}\n"
    )
    (info_dir / f"{dest.name}.trashinfo").write_text(info, encoding="utf-8")


def _unique_name(folder: Path, name: str) -> Path:
    dest = folder / name
    if not dest.exists():
        return dest
    stem = Path(name).stem
    suffix = Path(name).suffix
    n = 1
    while True:
        candidate = folder / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1
