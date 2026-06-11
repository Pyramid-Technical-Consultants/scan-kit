from pathlib import Path

import pytest
from PySide6.QtGui import QGuiApplication

from scan_kit.common.app_icon import asset_path, load_app_icon
from scan_kit.common.win_identity import prepare_windows_app_identity


@pytest.fixture(scope="module")
def qapp():
    app = QGuiApplication.instance()
    if app is None:
        app = QGuiApplication([])
    return app


def test_asset_path_resolves_bundled_icon_files() -> None:
    png = asset_path("icon.png")
    ico = asset_path("icon.ico")
    svg = asset_path("scan-kit-icon.svg")
    assert png.is_file()
    assert ico.is_file()
    assert svg.is_file()
    assert png.parent == Path(__file__).resolve().parent.parent / "scan_kit" / "assets"


def test_load_app_icon_finds_png(qapp) -> None:
    icon = load_app_icon()
    assert not icon.isNull()


def test_prepare_windows_app_identity_does_not_raise() -> None:
    prepare_windows_app_identity()
