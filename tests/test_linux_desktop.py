from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from scan_kit.common.linux_desktop import (
    _render_desktop_entry,
    ensure_linux_desktop_integration,
    should_install_linux_desktop,
)


def test_should_install_linux_desktop_only_for_main_frozen_process(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "argv", ["scan-kit"])
    assert should_install_linux_desktop() is True

    monkeypatch.setattr(sys, "argv", ["scan-kit", "--run-view", "position_scatter"])
    assert should_install_linux_desktop() is False


def test_should_install_linux_desktop_skips_non_linux(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "argv", ["scan-kit"])
    assert should_install_linux_desktop() is False


def test_render_desktop_entry_uses_absolute_exec() -> None:
    entry = _render_desktop_entry(Path("/opt/scan-kit/scan-kit"))
    assert "Exec=/opt/scan-kit/scan-kit" in entry
    assert "StartupWMClass=scan-kit" in entry
    assert "Icon=scan-kit" in entry


def test_ensure_linux_desktop_integration_installs_files(
    monkeypatch, tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    icon_src = assets / "icon.png"
    icon_src.write_bytes(b"png")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "argv", ["scan-kit"])
    monkeypatch.setattr(sys, "executable", str(tmp_path / "scan-kit"), raising=False)
    home = tmp_path / "home"
    monkeypatch.setattr(
        Path,
        "home",
        classmethod(lambda cls: home),
    )
    monkeypatch.setattr(
        "scan_kit.common.linux_desktop.asset_path",
        lambda name: assets / name,
    )
    monkeypatch.setattr("scan_kit.common.linux_desktop._refresh_desktop_database", lambda: None)

    ensure_linux_desktop_integration()

    icon_dest = home / ".local/share/icons/hicolor/256x256/apps/scan-kit.png"
    desktop_dest = home / ".local/share/applications/scan-kit.desktop"
    assert icon_dest.is_file()
    assert desktop_dest.is_file()
    assert f"Exec={Path(sys.executable).resolve().as_posix()}" in desktop_dest.read_text(encoding="utf-8")
    if os.name == "posix":
        assert oct(desktop_dest.stat().st_mode & 0o777) == oct(0o755)
