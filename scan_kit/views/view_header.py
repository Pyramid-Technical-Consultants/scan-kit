"""The strip above a configurable view: a note, then the view's buttons, then its picker in the top right."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QToolButton, QWidget


class ViewHeader(QWidget):
    """Pick what a view shows from *choices* ``[(data, text)]``; :meth:`add_button` puts buttons left of the picker."""

    picked = Signal(object)

    def __init__(self, choices: Sequence[tuple[object, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(4, 2, 4, 0)
        self.note = QLabel("")
        self.note.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.picker = QComboBox()
        for data, text in choices:
            self.picker.addItem(text, data)
        self.picker.currentIndexChanged.connect(lambda _i: self.picked.emit(self.current))
        self._row.addWidget(self.note, 1)
        self._row.addWidget(self.picker)

    @property
    def current(self):
        return self.picker.currentData()

    def set_current(self, data, *, quiet: bool = False) -> None:
        """Select *data* (else the first choice); *quiet* skips :attr:`picked`."""
        self.picker.blockSignals(quiet)
        self.picker.setCurrentIndex(max(self.picker.findData(data), 0))
        self.picker.blockSignals(False)

    def add_button(self, text: str, tooltip: str = "") -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        self._row.insertWidget(self._row.indexOf(self.picker), button)
        return button
