"""Tests for shared Qt widget helpers."""

from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QScrollArea, QSplitter, QVBoxLayout, QWidget

from scan_kit.common.qt_widgets import (
    configure_pane_scroll_area,
    configure_scroll_widget,
    make_pane_scroll_area,
    pane_window_color,
    set_pane_scroll_widget,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_pane_window_color_uses_parent_window(qapp) -> None:
    splitter = QSplitter()
    host = QWidget(splitter)
    assert pane_window_color(host) == host.parentWidget().palette().color(
        QPalette.ColorRole.Window
    )


def test_configure_pane_scroll_area_aligns_host_and_viewport(qapp) -> None:
    splitter = QSplitter()
    host = QWidget(splitter)
    layout = QVBoxLayout(host)
    scroll = QScrollArea(host)
    layout.addWidget(scroll)

    configure_pane_scroll_area(scroll, host=host)

    window = pane_window_color(host)
    assert host.palette().color(QPalette.ColorRole.Base) == window
    assert scroll.palette().color(QPalette.ColorRole.Base) == window
    assert scroll.viewport().palette().color(QPalette.ColorRole.Base) == window
    assert not scroll.viewport().autoFillBackground()


def test_set_pane_scroll_widget_disables_content_autofill(qapp) -> None:
    splitter = QSplitter()
    host = QWidget(splitter)
    scroll = make_pane_scroll_area()
    content = QWidget()

    set_pane_scroll_widget(scroll, content, host=host)

    assert scroll.widget() is content
    assert not content.autoFillBackground()
    assert content.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


def test_configure_scroll_widget_reapplies_after_parenting(qapp) -> None:
    scroll = make_pane_scroll_area()
    content = QWidget()
    scroll.setWidget(content)

    configure_scroll_widget(content)

    assert not content.autoFillBackground()
