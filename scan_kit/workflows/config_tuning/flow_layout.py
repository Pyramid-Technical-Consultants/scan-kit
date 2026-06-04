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
        self.setMinimumWidth(0)

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

    def _single_row_width(self) -> int:
        if not self._chips:
            return 0
        total = 0
        visible = 0
        for chip in self._chips:
            if not chip.isVisible():
                continue
            if visible:
                total += self._h_spacing
            total += self._chip_width(chip)
            visible += 1
        return total

    def _available_width(self) -> int:
        width = self.width()
        if width > 0:
            return width
        parent = self.parentWidget()
        while parent is not None:
            parent_width = parent.width()
            if parent_width > 0:
                return parent_width
            parent = parent.parentWidget()
        return 0

    def minimumSizeHint(self) -> QSize:
        layout_width = self._available_width()
        if layout_width <= 0:
            layout_width = self._single_row_width()
        height = self._layout_height(layout_width)
        return QSize(0, max(height, 0))

    def heightForWidth(self, width: int) -> int:
        return self._layout_height(width)

    def hasHeightForWidth(self) -> bool:
        return True

    def _update_height(self) -> None:
        layout_width = self._available_width()
        if layout_width <= 0:
            layout_width = self._single_row_width()
        needed_height = self._layout_height(layout_width)
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
        width = self._available_width()
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
