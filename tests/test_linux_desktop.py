from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from scan_kit.common.app_icon import asset_path
from scan_kit.common.linux_desktop import (
    _frozen_launcher_path,
    _render_desktop_entry,
    ensure_linux_desktop_integration,
    install_appdir_icons,
    png_pixel_size,
    should_install_linux_desktop,
)


def test_should_install_linux_desktop_only_for_main_frozen_process(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "argv", ["scan-kit"])
    assert should_install_linux_desktop() is True

    monkeypatch.setattr(sys, "argv", ["scan-kit", "--run-view", "distribution"])
    assert should_install_linux_desktop() is False


def test_should_install_linux_desktop_skips_non_linux(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "argv", ["scan-kit"])
    assert should_install_linux_desktop() is False


def test_frozen_launcher_path_prefers_appimage_env(
    monkeypatch, tmp_path: Path,
) -> None:
    appimage = tmp_path / "Scan-Kit.AppImage"
    appimage.write_text("", encoding="utf-8")
    monkeypatch.setenv("APPIMAGE", str(appimage))
    assert _frozen_launcher_path() == appimage


def test_render_desktop_entry_uses_absolute_exec() -> None:
    exe = Path("/opt/scan-kit/scan-kit")
    entry = _render_desktop_entry(exe)
    assert f"Exec={exe.resolve().as_posix()}" in entry
    assert f"Path={exe.resolve().parent.as_posix()}" in entry
    assert "StartupWMClass=scan-kit" in entry
    assert "Icon=scan-kit" in entry
    assert "Keywords=proton;beam;scanning;dosimetry;IC;" in entry


def test_ensure_linux_desktop_integration_installs_files(
    monkeypatch, tmp_path: Path,
) -> None:
    icon_src = asset_path("icon.png")
    stale_home_icon = tmp_path / "home/.local/share/icons/hicolor/16x16/apps/scan-kit.png"

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "argv", ["scan-kit"])
    monkeypatch.setattr(sys, "executable", str(tmp_path / "scan-kit"), raising=False)
    home = tmp_path / "home"
    stale_home_icon.parent.mkdir(parents=True)
    stale_home_icon.write_bytes(b"stale")
    monkeypatch.setattr(
        Path,
        "home",
        classmethod(lambda cls: home),
    )
    monkeypatch.setattr(
        "scan_kit.common.linux_desktop.asset_path",
        lambda name: icon_src if name == "icon.png" else tmp_path / name,
    )
    monkeypatch.setattr("scan_kit.common.linux_desktop._refresh_desktop_database", lambda: None)
    monkeypatch.setattr("scan_kit.common.linux_desktop._refresh_icon_cache", lambda _root: None)

    ensure_linux_desktop_integration()

    icon_root = home / ".local/share/icons/hicolor"
    desktop_dest = home / ".local/share/applications/scan-kit.desktop"
    installed = icon_root / "256x256/apps/scan-kit.png"
    assert installed.is_file()
    assert png_pixel_size(installed) == (256, 256)
    assert not (icon_root / "16x16/apps/scan-kit.png").exists()
    assert not (icon_root / "scalable/apps/scan-kit.svg").exists()
    assert desktop_dest.is_file()
    assert (
        f"Exec={Path(sys.executable).resolve().as_posix()}"
        in desktop_dest.read_text(encoding="utf-8")
    )
    if os.name == "posix":
        assert oct(desktop_dest.stat().st_mode & 0o777) == oct(0o755)


def test_install_appdir_icons_writes_real_diricon(tmp_path: Path) -> None:
    icon_src = asset_path("icon.png")
    appdir = tmp_path / "ScanKit.AppDir"
    appdir.mkdir()
    stale = appdir / ".DirIcon"
    try:
        stale.symlink_to("missing.png")
    except OSError:
        stale.write_bytes(b"stale-diricon")

    install_appdir_icons(appdir, icon_src)

    diricon = appdir / ".DirIcon"
    root_png = appdir / "scan-kit.png"
    native = appdir / "usr/share/icons/hicolor/256x256/apps/scan-kit.png"
    assert diricon.is_file()
    assert not diricon.is_symlink()
    assert diricon.read_bytes() == icon_src.read_bytes()
    assert root_png.read_bytes() == icon_src.read_bytes()
    assert png_pixel_size(native) == (256, 256)
    assert not (appdir / "usr/share/icons/hicolor/scalable/apps/scan-kit.svg").exists()
    extra = appdir / "usr/share/icons/hicolor/48x48/apps/scan-kit.png"
    if extra.is_file():
        assert png_pixel_size(extra) == (48, 48)


def test_packaging_desktop_file_uses_scan_kit_icon() -> None:
    packaging = Path(__file__).resolve().parent.parent / "packaging/linux"
    desktop = packaging / "appimage.desktop"
    text = desktop.read_text(encoding="utf-8")
    assert "Icon=scan-kit" in text
    assert "StartupWMClass=scan-kit" in text
    assert "Exec=scan-kit" in text
    metainfo = packaging / "scan-kit.appdata.xml"
    assert metainfo.is_file()
    xml = metainfo.read_text(encoding="utf-8")
    assert '<icon type="stock">scan-kit</icon>' in xml
    assert "scan-kit.desktop" in xml
    script = packaging / "build_appimage.sh"
    body = script.read_text(encoding="utf-8")
    assert "install_appdir_icons" in body
    assert "ln -sf scan-kit.png" not in body
