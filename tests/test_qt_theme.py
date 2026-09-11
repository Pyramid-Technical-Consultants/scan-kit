"""Tests for Qt light/dark/system color scheme helpers."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QMenu, QStyle, QWidget

from scan_kit.common.app_settings import AppSettings, normalize_ui_theme
from scan_kit.common.qt_theme import (
    add_theme_menu,
    apply_ui_theme,
    persist_ui_theme,
    tinted_standard_icon,
)


def test_normalize_ui_theme() -> None:
    assert normalize_ui_theme("Dark") == "dark"
    assert normalize_ui_theme("light") == "light"
    assert normalize_ui_theme("system") == "system"
    assert normalize_ui_theme("neon") == "system"
    assert normalize_ui_theme(None) == "system"


def test_apply_ui_theme_sets_color_scheme(qapp) -> None:
    apply_ui_theme("dark", app=qapp)
    assert qapp.styleHints().colorScheme() == Qt.ColorScheme.Dark
    apply_ui_theme("light", app=qapp)
    assert qapp.styleHints().colorScheme() == Qt.ColorScheme.Light
    apply_ui_theme("system", app=qapp)
    # Qt reports the resolved OS scheme, not ColorScheme.Unknown.
    assert qapp.styleHints().colorScheme() in (
        Qt.ColorScheme.Light,
        Qt.ColorScheme.Dark,
        Qt.ColorScheme.Unknown,
    )


def test_tinted_standard_icon_is_usable(qapp) -> None:
    widget = QWidget()
    icon = tinted_standard_icon(widget, QStyle.StandardPixmap.SP_MediaPlay)
    assert not icon.isNull()
    widget.deleteLater()


def test_persist_ui_theme_round_trip(qapp, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scan_kit.common.app_settings._SETTINGS_DIR", tmp_path)
    persist_ui_theme("dark")
    assert AppSettings.load().ui_theme == "dark"
    persist_ui_theme("light")
    assert AppSettings.load().ui_theme == "light"


def test_theme_menu_applies_only_checked_action(
    qapp, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scan_kit.common.app_settings._SETTINGS_DIR", tmp_path)
    calls: list[str] = []

    def _capture(theme: str, *, settings=None) -> None:
        calls.append(theme)

    monkeypatch.setattr("scan_kit.common.qt_theme.persist_ui_theme", _capture)
    window = QMainWindow()
    menu = QMenu(window)
    add_theme_menu(window, menu)
    actions = {action.text(): action for action in menu.actions()}
    actions["Dark"].trigger()
    assert calls == ["dark"]
    window.close()
