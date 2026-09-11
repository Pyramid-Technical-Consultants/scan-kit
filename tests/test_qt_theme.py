"""Tests for Qt light/dark/system color scheme helpers."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QStyle, QWidget

from scan_kit.common.app_settings import normalize_ui_theme
from scan_kit.common.qt_theme import apply_ui_theme, tinted_standard_icon


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
