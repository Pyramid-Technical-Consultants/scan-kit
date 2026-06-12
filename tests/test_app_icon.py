from pathlib import Path

import pytest
from PySide6.QtGui import QGuiApplication

from scan_kit.common.app_icon import (
    apply_qt_application_branding,
    asset_path,
    desktop_file_name,
    load_app_icon,
    prepare_qt_app_identity,
)
from scan_kit.common.linux_frozen_env import prepare_linux_frozen_env
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


def test_prepare_qt_app_identity_sets_stable_names(qapp) -> None:
    from PySide6.QtCore import QCoreApplication

    prepare_qt_app_identity()
    assert QCoreApplication.organizationName() == "ProtonCare"
    assert QCoreApplication.applicationName() == "scan-kit"


def test_desktop_file_name_matches_wm_class() -> None:
    assert desktop_file_name() == "scan-kit"


def test_apply_qt_application_branding_sets_icon(qapp) -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    assert app is not None
    icon = apply_qt_application_branding(app)
    assert not icon.isNull()
    assert not app.windowIcon().isNull()


def test_prepare_windows_app_identity_does_not_raise() -> None:
    prepare_windows_app_identity()


def test_prepare_linux_frozen_env_noop_when_not_frozen(monkeypatch) -> None:
    monkeypatch.delenv("GIO_MODULE_DIR", raising=False)
    prepare_linux_frozen_env()
    assert "GIO_MODULE_DIR" not in __import__("os").environ


def test_prepare_linux_frozen_env_sets_isolation_vars(monkeypatch) -> None:
    import os
    import sys

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delenv("GIO_MODULE_DIR", raising=False)
    monkeypatch.delenv("QT_IM_MODULE", raising=False)

    prepare_linux_frozen_env()

    assert os.environ["GIO_MODULE_DIR"] == ""
    assert os.environ["QT_IM_MODULE"] == "simple"
    assert os.environ["NO_AT_BRIDGE"] == "1"
