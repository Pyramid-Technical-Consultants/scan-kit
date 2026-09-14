"""Register scan-kit with the Linux desktop shell on first launch."""

from __future__ import annotations

import os
import shutil
import struct
import sys
from pathlib import Path

from .app_icon import asset_path, desktop_file_name

_ICON_THEME_NAME = desktop_file_name()
_ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
_FALLBACK_ICON_SIZE = 256
_SUBPROCESS_FLAGS = frozenset({"--run-view", "--warm-worker", "--version", "-V"})


def should_install_linux_desktop() -> bool:
    """Return whether this process should update the user desktop entry."""
    if sys.platform != "linux" or not getattr(sys, "frozen", False):
        return False
    return not any(flag in sys.argv for flag in _SUBPROCESS_FLAGS)


def _desktop_entry_path() -> Path:
    return Path.home() / ".local" / "share" / "applications" / f"{_ICON_THEME_NAME}.desktop"


def _hicolor_icons_root() -> Path:
    return Path.home() / ".local" / "share" / "icons" / "hicolor"


def png_pixel_size(path: Path) -> tuple[int, int] | None:
    """Return ``(width, height)`` for a PNG, or ``None`` if it is not a PNG."""
    try:
        with path.open("rb") as fh:
            header = fh.read(24)
    except OSError:
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        return None
    return width, height


def _hicolor_png_path(root: Path, size: int, name: str) -> Path:
    return root / f"{size}x{size}" / "apps" / f"{name}.png"


def _copy_png(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _write_resized_hicolor_pngs(root: Path, icon_src: Path, name: str, native: int) -> None:
    """Write extra hicolor sizes. Best-effort; skipped when Qt cannot load the PNG."""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage

        image = QImage(str(icon_src))
        if image.isNull():
            return
        for size in _ICON_SIZES:
            if size == native:
                continue
            dest = _hicolor_png_path(root, size, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            scaled = image.scaled(
                size,
                size,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            scaled.save(str(dest), "PNG")
    except Exception:
        return


def install_hicolor_png(
    root: Path,
    icon_src: Path,
    *,
    name: str = _ICON_THEME_NAME,
    extra_sizes: bool = False,
) -> int:
    """Install *icon_src* into a hicolor tree at its native pixel size.

    Returns the native size used. Stale copies in other size slots (and an
    Inkscape scalable SVG, which many theme loaders fail to rasterize) are
    removed so ``Icon=scan-kit`` resolves to a real PNG.
    """
    size_info = png_pixel_size(icon_src)
    native = size_info[0] if size_info and size_info[0] == size_info[1] else _FALLBACK_ICON_SIZE
    _copy_png(icon_src, _hicolor_png_path(root, native, name))
    scalable = root / "scalable" / "apps" / f"{name}.svg"
    if scalable.is_file() or scalable.is_symlink():
        scalable.unlink()
    if extra_sizes:
        _write_resized_hicolor_pngs(root, icon_src, name, native)
    else:
        for size in _ICON_SIZES:
            if size == native:
                continue
            stale = _hicolor_png_path(root, size, name)
            if stale.is_file() or stale.is_symlink():
                stale.unlink()
    return native


def install_appdir_icons(appdir: Path, icon_src: Path) -> None:
    """Install AppDir root icon, a real ``.DirIcon``, and hicolor theme icons.

    ``.DirIcon`` must be a regular file: AppImage thumbnailers extract it
    alone, so a symlink to ``scan-kit.png`` becomes a dangling shortcut.
    """
    if not icon_src.is_file():
        raise FileNotFoundError(icon_src)
    appdir.mkdir(parents=True, exist_ok=True)
    root_png = appdir / f"{_ICON_THEME_NAME}.png"
    shutil.copy2(icon_src, root_png)
    diricon = appdir / ".DirIcon"
    if diricon.exists() or diricon.is_symlink():
        diricon.unlink()
    shutil.copy2(icon_src, diricon)
    install_hicolor_png(
        appdir / "usr" / "share" / "icons" / "hicolor",
        icon_src,
        extra_sizes=True,
    )


def _install_hicolor_icons(icon_src: Path) -> bool:
    """Install the user-theme PNG so ``Icon=scan-kit`` resolves after first launch."""
    root = _hicolor_icons_root()
    try:
        install_hicolor_png(root, icon_src, extra_sizes=False)
    except OSError:
        return False
    _refresh_icon_cache(root)
    return True


def _frozen_launcher_path() -> Path:
    """Path users should launch (AppImage file when running from an AppImage)."""
    appimage = os.environ.get("APPIMAGE", "").strip()
    if appimage:
        return Path(appimage)
    return Path(sys.executable).resolve()


def _render_desktop_entry(exe: Path) -> str:
    exe_path = exe.resolve()
    app_dir = exe_path.parent
    return "\n".join(
        (
            "[Desktop Entry]",
            "Type=Application",
            "Name=Scan Kit",
            "GenericName=Scan Kit",
            "Comment=Proton pencil beam scanning analysis toolkit",
            f"Exec={exe_path.as_posix()}",
            f"Path={app_dir.as_posix()}",
            f"Icon={_ICON_THEME_NAME}",
            "Terminal=false",
            "StartupNotify=true",
            "Categories=Science;Utility;",
            "Keywords=proton;beam;scanning;dosimetry;IC;",
            f"StartupWMClass={_ICON_THEME_NAME}",
            "",
        )
    )


def _needs_desktop_refresh(exe: Path) -> bool:
    desktop_path = _desktop_entry_path()
    if not desktop_path.is_file():
        return True
    try:
        return f"Exec={exe.resolve().as_posix()}" not in desktop_path.read_text(
            encoding="utf-8",
        )
    except OSError:
        return True


def ensure_linux_desktop_integration() -> None:
    """Install or refresh the user ``.desktop`` entry and hicolor icon."""
    if not should_install_linux_desktop():
        return

    icon_src = asset_path("icon.png")
    if not icon_src.is_file():
        return

    exe = _frozen_launcher_path()
    desktop_dest = _desktop_entry_path()

    try:
        desktop_dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    if not _install_hicolor_icons(icon_src):
        return

    if _needs_desktop_refresh(exe):
        try:
            desktop_dest.write_text(_render_desktop_entry(exe), encoding="utf-8")
            os.chmod(desktop_dest, 0o755)
        except OSError:
            return
        _refresh_desktop_database()


def _refresh_icon_cache(hicolor_dir: Path) -> None:
    try:
        from shutil import which

        updater = which("gtk-update-icon-cache")
        if updater:
            import subprocess

            subprocess.run(
                [updater, "-f", "-t", str(hicolor_dir)],
                check=False,
                timeout=2,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass


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
                timeout=2,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass
