"""Register scan-kit with the Linux desktop shell on first launch."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .app_icon import asset_path, desktop_file_name

_ICON_THEME_NAME = desktop_file_name()
_SUBPROCESS_FLAGS = frozenset({"--run-view", "--warm-worker", "--version", "-V"})


def should_install_linux_desktop() -> bool:
    """Return whether this process should update the user desktop entry."""
    if sys.platform != "linux" or not getattr(sys, "frozen", False):
        return False
    return not any(flag in sys.argv for flag in _SUBPROCESS_FLAGS)


def _desktop_entry_path() -> Path:
    return Path.home() / ".local" / "share" / "applications" / f"{_ICON_THEME_NAME}.desktop"


def _icon_path() -> Path:
    return (
        Path.home()
        / ".local"
        / "share"
        / "icons"
        / "hicolor"
        / "256x256"
        / "apps"
        / f"{_ICON_THEME_NAME}.png"
    )


def _render_desktop_entry(exe: Path) -> str:
    return "\n".join(
        (
            "[Desktop Entry]",
            "Type=Application",
            "Name=Scan Kit",
            "GenericName=Scan Kit",
            "Comment=Proton pencil beam scanning analysis toolkit",
            f"Exec={exe.as_posix()}",
            f"Icon={_ICON_THEME_NAME}",
            "Terminal=false",
            "Categories=Science;Utility;",
            f"StartupWMClass={_ICON_THEME_NAME}",
            "",
        )
    )


def _needs_desktop_refresh(exe: Path) -> bool:
    desktop_path = _desktop_entry_path()
    if not desktop_path.is_file():
        return True
    try:
        return f"Exec={exe}" not in desktop_path.read_text(encoding="utf-8")
    except OSError:
        return True


def ensure_linux_desktop_integration() -> None:
    """Install or refresh the user ``.desktop`` entry and hicolor icon."""
    if not should_install_linux_desktop():
        return

    icon_src = asset_path("icon.png")
    if not icon_src.is_file():
        return

    exe = Path(sys.executable).resolve()
    icon_dest = _icon_path()
    desktop_dest = _desktop_entry_path()

    try:
        icon_dest.parent.mkdir(parents=True, exist_ok=True)
        desktop_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(icon_src, icon_dest)
    except OSError:
        return

    if _needs_desktop_refresh(exe):
        try:
            desktop_dest.write_text(_render_desktop_entry(exe), encoding="utf-8")
            os.chmod(desktop_dest, 0o755)
        except OSError:
            return

    _refresh_desktop_database()


def _refresh_desktop_database() -> None:
    apps_dir = _desktop_entry_path().parent
    try:
        from shutil import which

        updater = which("update-desktop-database")
        if updater:
            import subprocess

            subprocess.run(
                [updater, str(apps_dir)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass
