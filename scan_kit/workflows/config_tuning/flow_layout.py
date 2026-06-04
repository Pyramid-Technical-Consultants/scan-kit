"""Wrapping horizontal layout for inline attribute chips."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QSizePolicy, QWidget


class FlowWidget(QWidget):
    """Lay out child widgets left-to-right, wrapping to the next row as needed."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        h_spacing: int = 8,
        v_spacing: int = 6,
    ) -> None:
        super().__init__(parent)
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self._chips: list[QWidget] = []
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def addWidget(self, widget: QWidget) -> None:
        widget.setParent(self)
        self._chips.append(widget)
        self._update_height()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._relayout()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._relayout()

    def sizeHint(self) -> QSize:
        return self.minimumSizeHint()

    def minimumSizeHint(self) -> QSize:
        width = self._effective_width()
        height = self._layout_height(width)
        return QSize(0, max(height, 0))

    def heightForWidth(self, width: int) -> int:
        return self._layout_height(width)

    def hasHeightForWidth(self) -> bool:
        return True

    def _effective_width(self) -> int:
        if self.width() > 0:
            return self.width()
        parent = self.parentWidget()
        if parent is not None and parent.width() > 0:
            return parent.width()
        return 640

    def _update_height(self) -> None:
        needed_height = self._layout_height(self._effective_width())
        if needed_height > 0 and self.minimumHeight() != needed_height:
            self.setMinimumHeight(needed_height)
            self.updateGeometry()

    def _chip_width(self, chip: QWidget) -> int:
        return max(
            chip.minimumWidth(),
            chip.minimumSizeHint().width(),
            chip.sizeHint().width(),
        )

    def _chip_height(self, chip: QWidget) -> int:
        return max(
            chip.minimumHeight(),
            chip.minimumSizeHint().height(),
            chip.sizeHint().height(),
        )

    def _layout_height(self, width: int) -> int:
        if width <= 0 or not self._chips:
            return 0

        x = 0
        y = 0
        line_height = 0
        for chip in self._chips:
            if not chip.isVisible():
                continue
            chip_width = self._chip_width(chip)
            chip_height = self._chip_height(chip)
            if x > 0 and x + chip_width > width:
                x = 0
                y += line_height + self._v_spacing
                line_height = 0
            x += chip_width + self._h_spacing
            line_height = max(line_height, chip_height)
        return y + line_height

    def _relayout(self) -> None:
        width = self.width()
        if width <= 0:
            return

        x = 0
        y = 0
        line_height = 0
        for chip in self._chips:
            if not chip.isVisible():
                continue
            chip_width = self._chip_width(chip)
            chip_height = self._chip_height(chip)
            if x > 0 and x + chip_width > width:
                x = 0
                y += line_height + self._v_spacing
                line_height = 0
            chip.setGeometry(x, y, chip_width, chip_height)
            chip.show()
            x += chip_width + self._h_spacing
            line_height = max(line_height, chip_height)

        needed_height = y + line_height
        if self.minimumHeight() != needed_height:
            self.setMinimumHeight(needed_height)
            self.updateGeometry()
