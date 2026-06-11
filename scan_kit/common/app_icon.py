"""Application icon and bundled asset paths."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

from PySide6.QtGui import QIcon

from .win_identity import prepare_windows_app_identity

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_ASSETS_DIR = _PACKAGE_DIR / "assets"
_SVG_NAME = "scan-kit-icon.svg"
_PNG_NAME = "icon.png"
_ICO_NAME = "icon.ico"

__all__ = [
    "apply_windows_window_icons",
    "asset_path",
    "load_app_icon",
    "prepare_windows_app_identity",
]


def asset_path(name: str) -> Path:
    """Return the path to a file under ``scan_kit/assets/``."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "scan_kit" / "assets" / name
    return _ASSETS_DIR / name


def load_app_icon() -> QIcon:
    """Load the scan-kit window icon from bundled assets."""
    if sys.platform == "win32":
        ico = asset_path(_ICO_NAME)
        if ico.is_file():
            icon = QIcon(str(ico))
            if not icon.isNull():
                return icon

    png = asset_path(_PNG_NAME)
    if png.is_file():
        return QIcon(str(png))

    svg = asset_path(_SVG_NAME)
    if svg.is_file():
        return QIcon(str(svg))
    return QIcon()


def apply_windows_window_icons(window, icon: QIcon | None = None) -> None:
    """Push icon handles to the native window for the Windows taskbar."""
    if sys.platform != "win32":
        return
    if icon is not None and icon.isNull():
        return

    ico_path = asset_path(_ICO_NAME)
    if not ico_path.is_file():
        return

    try:
        hwnd = int(window.winId())
    except (AttributeError, TypeError, ValueError):
        return
    if not hwnd:
        return

    LR_LOADFROMFILE = 0x0010
    IMAGE_ICON = 1
    WM_SETICON = 0x0080
    ICON_SMALL = 0
    ICON_BIG = 1
    path = str(ico_path.resolve())

    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        for size, icon_type in ((16, ICON_SMALL), (32, ICON_BIG), (48, ICON_BIG)):
            handle = user32.LoadImageW(None, path, IMAGE_ICON, size, size, LR_LOADFROMFILE)
            if handle:
                user32.SendMessageW(hwnd, WM_SETICON, icon_type, handle)
    except Exception:
        pass
