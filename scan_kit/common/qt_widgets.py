"""Shared Qt widget helpers used across ScanKit panels."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QFrame, QScrollArea, QWidget


def pane_window_color(widget: QWidget) -> QColor:
    """Return the Window palette color for *widget*'s parent pane."""
    parent = widget.parentWidget()
    if parent is not None:
        return parent.palette().color(QPalette.ColorRole.Window)
    return widget.palette().color(QPalette.ColorRole.Window)


def blend_widget_with_pane(
    widget: QWidget,
    *,
    window_color: QColor | None = None,
) -> QColor:
    """Disable auto-fill and match *widget* Base to the pane Window color."""
    if window_color is None:
        window_color = pane_window_color(widget)
    widget.setAutoFillBackground(False)
    pal = widget.palette()
    pal.setColor(QPalette.ColorRole.Base, window_color)
    widget.setPalette(pal)
    return window_color


def configure_scroll_widget(
    content: QWidget,
    *,
    window_color: QColor | None = None,
) -> None:
    """Apply scroll-content background rules after :meth:`QScrollArea.setWidget`."""
    if window_color is None:
        parent = content.parentWidget()
        if parent is not None:
            window_color = pane_window_color(parent)
        else:
            window_color = content.palette().color(QPalette.ColorRole.Window)
    content.setAutoFillBackground(False)
    content.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    blend_widget_with_pane(content, window_color=window_color)


def configure_pane_scroll_area(
    scroll: QScrollArea,
    *,
    host: QWidget | None = None,
    content: QWidget | None = None,
) -> None:
    """Blend a scroll area (and optional host/content) with its parent pane."""
    anchor = host or scroll
    window_color = pane_window_color(anchor)
    widgets: list[QWidget] = []
    if host is not None:
        widgets.append(host)
    widgets.extend((scroll, scroll.viewport()))
    for widget in widgets:
        blend_widget_with_pane(widget, window_color=window_color)
    if content is not None:
        configure_scroll_widget(content, window_color=window_color)


def make_pane_scroll_area(
    *,
    widget_resizable: bool = True,
    frame_shape: QFrame.Shape = QFrame.Shape.NoFrame,
) -> QScrollArea:
    """Create a :class:`QScrollArea` that blends with its parent splitter pane."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(widget_resizable)
    scroll.setFrameShape(frame_shape)
    configure_pane_scroll_area(scroll)
    return scroll


def set_pane_scroll_widget(
    scroll: QScrollArea,
    content: QWidget,
    *,
    host: QWidget | None = None,
) -> None:
    """Set scroll content and apply pane background rules to the full chain."""
    scroll.setWidget(content)
    configure_pane_scroll_area(scroll, host=host, content=content)
