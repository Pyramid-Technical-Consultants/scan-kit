"""Tests for the main-window menu bar."""

from __future__ import annotations

import sys

import pytest
from PySide6.QtWidgets import QApplication, QMenu

from scan_kit.qt_launcher import (
    _MAIN_TAB_CONFIG_TUNING,
    _MAIN_TAB_DATA_ANALYSIS,
    _MAIN_TAB_DEBUG,
    ScanKitMainWindow,
)
from scan_kit.views import VIEW_GROUPS


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def window(qapp):
    win = ScanKitMainWindow()
    win._build_ui()  # bypass the deferred warm-pool init
    try:
        yield win
    finally:
        win._shutdown_children()
        win.close()


def _menu(win: ScanKitMainWindow, title: str) -> QMenu:
    # Use the window's strong references; re-deriving via QAction.menu() can hit a
    # PySide6 wrapper-ownership quirk where the menu wrapper is invalidated.
    for menu in win._menus:
        if menu.title().replace("&", "") == title:
            return menu
    raise AssertionError(f"menu {title!r} not found")


def test_top_level_menus_present(window) -> None:
    titles = [m.title().replace("&", "") for m in window._menus]
    assert titles == ["File", "Edit", "View", "Analysis", "Help"]


def test_edit_menu_uses_browser_undo_actions(window) -> None:
    edit = _menu(window, "Edit")
    actions = edit.actions()
    assert window._session_browser is not None
    assert window._session_browser.undo_action() in actions
    assert window._session_browser.redo_action() in actions


def test_analysis_menu_has_submenu_per_view_group(window) -> None:
    analysis = _menu(window, "Analysis")
    submenu_titles = [a.text() for a in analysis.actions() if a.menu() is not None]
    group_titles = [title for title, _ in VIEW_GROUPS]
    # Every view group has a submenu; Calibration is the one extra submenu.
    for title in group_titles:
        assert title in submenu_titles
    assert "Calibration" in submenu_titles


def test_view_menu_switches_tab_and_syncs(window) -> None:
    window._switch_to_main_tab(_MAIN_TAB_DATA_ANALYSIS)
    window._sync_tab_menu()
    assert window._tab_menu_actions[_MAIN_TAB_DATA_ANALYSIS].isChecked()

    window._tab_menu_actions[_MAIN_TAB_CONFIG_TUNING].trigger()
    tabs = window._main_tabs
    assert tabs.tabText(tabs.currentIndex()) == _MAIN_TAB_CONFIG_TUNING
    assert window._tab_menu_actions[_MAIN_TAB_CONFIG_TUNING].isChecked()
    assert not window._tab_menu_actions[_MAIN_TAB_DEBUG].isChecked()


def test_background_subtraction_action_syncs_with_settings(window) -> None:
    window._settings.bg_subtract = True
    window._sync_bg_buttons()
    assert window._bg_menu_action.isChecked()

    window._settings.bg_subtract = False
    window._sync_bg_buttons()
    assert not window._bg_menu_action.isChecked()


def test_calibration_actions_form_exclusive_radio(window) -> None:
    window._settings.calibration_mode = "per_session"
    window._sync_cal_buttons()
    checked = [m for m, a in window._cal_menu_actions.items() if a.isChecked()]
    assert checked == ["per_session"]
