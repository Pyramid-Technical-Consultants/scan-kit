"""Collapsible fieldset widget for the configuration tuning form."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleGroupBox(QWidget):
    """A titled, collapsible section similar to a ``QGroupBox``."""

    def __init__(
        self,
        title: str = "",
        parent: QWidget | None = None,
        *,
        expanded: bool = True,
    ) -> None:
        super().__init__(parent)
        self._title_text = title
        self._expanded = expanded

        self._content = QWidget()
        self._content.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = QFrame()
        self._header.setFrameShape(QFrame.Shape.NoFrame)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        header_row = QHBoxLayout(self._header)
        header_row.setContentsMargins(4, 6, 4, 6)
        header_row.setSpacing(6)

        self._toggle = QToolButton()
        self._toggle.setAutoRaise(True)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._toggle.clicked.connect(self._toggle_expanded)

        self._title_label = QLabel(title)
        title_font = self._title_label.font()
        title_font.setBold(True)
        self._title_label.setFont(title_font)

        header_row.addWidget(self._toggle)
        header_row.addWidget(self._title_label, stretch=1)

        self._body = QFrame()
        self._body.setFrameShape(QFrame.Shape.StyledPanel)
        body_outer = QVBoxLayout(self._body)
        body_outer.setContentsMargins(0, 0, 0, 0)
        body_outer.setSpacing(0)
        body_outer.addWidget(self._content)

        outer.addWidget(self._header)
        outer.addWidget(self._body)

        self._header.mousePressEvent = self._header_mouse_press  # type: ignore[method-assign]
        self.setExpanded(expanded)

    def title(self) -> str:
        return self._title_text

    def setTitle(self, title: str) -> None:  # noqa: N802 - Qt naming
        self._title_text = title
        self._title_label.setText(title)

    def isExpanded(self) -> bool:  # noqa: N802 - Qt naming
        return self._expanded

    def setExpanded(self, expanded: bool) -> None:  # noqa: N802 - Qt naming
        self._expanded = expanded
        self._body.setVisible(expanded)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.updateGeometry()

    def content_layout(
        self,
        *,
        margins: tuple[int, int, int, int] = (8, 8, 8, 8),
        spacing: int = 8,
    ) -> QVBoxLayout:
        """Create (or return) the layout used for section contents."""
        layout = self._content.layout()
        if layout is None:
            layout = QVBoxLayout(self._content)
            layout.setContentsMargins(*margins)
            layout.setSpacing(spacing)
        return layout

    def _toggle_expanded(self) -> None:
        self.setExpanded(not self._expanded)

    def _header_mouse_press(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_expanded()
        event.accept()


def make_collapsible_fieldset(
    title: str,
    *,
    expanded: bool = True,
    margins: tuple[int, int, int, int] = (8, 8, 8, 8),
    spacing: int = 8,
) -> tuple[CollapsibleGroupBox, QVBoxLayout]:
    """Return a collapsible fieldset and its content layout."""
    box = CollapsibleGroupBox(title, expanded=expanded)
    box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    box.setMinimumWidth(0)
    layout = box.content_layout(margins=margins, spacing=spacing)
    return box, layout


class PlainFieldset(QGroupBox):
    """Non-collapsible titled section for leafish form groups."""

    def content_layout(
        self,
        *,
        margins: tuple[int, int, int, int] = (8, 8, 8, 8),
        spacing: int = 8,
    ) -> QVBoxLayout:
        layout = self.layout()
        if layout is None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(*margins)
            layout.setSpacing(spacing)
        return layout


def make_plain_fieldset(
    title: str,
    *,
    margins: tuple[int, int, int, int] = (8, 8, 8, 8),
    spacing: int = 8,
) -> tuple[PlainFieldset, QVBoxLayout]:
    """Return a static ``QGroupBox`` fieldset and its content layout."""
    box = PlainFieldset(title)
    box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    box.setMinimumWidth(0)
    layout = box.content_layout(margins=margins, spacing=spacing)
    return box, layout


def make_form_section(
    title: str,
    *,
    collapsible: bool,
    expanded: bool = True,
    margins: tuple[int, int, int, int] = (8, 8, 8, 8),
    spacing: int = 8,
) -> tuple[CollapsibleGroupBox | PlainFieldset, QVBoxLayout]:
    """Return a collapsible or plain fieldset depending on *collapsible*."""
    if collapsible:
        return make_collapsible_fieldset(
            title,
            expanded=expanded,
            margins=margins,
            spacing=spacing,
        )
    return make_plain_fieldset(title, margins=margins, spacing=spacing)


def section_title(section: CollapsibleGroupBox | PlainFieldset) -> str:
    if isinstance(section, CollapsibleGroupBox):
        return section.title()
    return section.title()


def iter_form_sections(root: QWidget) -> list[CollapsibleGroupBox | PlainFieldset]:
    """Return all form section widgets under *root* in construction order."""
    sections: list[CollapsibleGroupBox | PlainFieldset] = []

    def walk(widget: QWidget) -> None:
        if isinstance(widget, (CollapsibleGroupBox, PlainFieldset)):
            sections.append(widget)
        for child in widget.children():
            if isinstance(child, QWidget):
                walk(child)

    walk(root)
    return sections
