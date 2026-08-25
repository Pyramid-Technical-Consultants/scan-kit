"""Compact segmented (radio-style) button groups for Qt side panels."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QWidget,
)

# Connected checkable buttons; selected segment uses palette highlight roles.
_SEGMENTED_QSS = """
SegmentedControl QPushButton {
    border: 1px solid palette(mid);
    padding: 5px 14px;
    background: palette(button);
    color: palette(button-text);
}
SegmentedControl QPushButton[seg="mid"],
SegmentedControl QPushButton[seg="right"] {
    border-left: none;
}
SegmentedControl QPushButton[seg="left"] {
    border-top-left-radius: 6px;
    border-bottom-left-radius: 6px;
}
SegmentedControl QPushButton[seg="right"] {
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}
SegmentedControl QPushButton[seg="only"] {
    border-radius: 6px;
}
SegmentedControl QPushButton:hover:!checked {
    background: palette(midlight);
}
SegmentedControl QPushButton:checked {
    background: palette(highlight);
    color: palette(highlighted-text);
    border-color: palette(highlight);
}
SegmentedControl QPushButton:disabled {
    color: palette(mid);
}
"""


class SegmentedControl(QWidget):
    """Horizontal selector: connected checkable buttons acting as one radio set.

    Build with ``[(key, label), …]``; emits :attr:`selectionChanged` with the
    chosen key on user interaction only. Use :meth:`set_current` to reflect
    external state without re-emitting.
    """

    selectionChanged = Signal(str)

    def __init__(
        self,
        options: list[tuple[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("SegmentedControl")
        self._buttons: dict[str, QPushButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        n = len(options)
        for i, (key, label) in enumerate(options):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            if n == 1:
                seg = "only"
            elif i == 0:
                seg = "left"
            elif i == n - 1:
                seg = "right"
            else:
                seg = "mid"
            btn.setProperty("seg", seg)
            self._group.addButton(btn)
            self._buttons[key] = btn
            lay.addWidget(btn)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(_SEGMENTED_QSS)
        self._group.buttonClicked.connect(self._on_clicked)

    def _on_clicked(self, button: QPushButton) -> None:
        for key, btn in self._buttons.items():
            if btn is button:
                self.selectionChanged.emit(key)
                return

    def set_current(self, key: str) -> None:
        btn = self._buttons.get(key)
        if btn is not None:
            btn.setChecked(True)

    def set_option_enabled(self, key: str, enabled: bool) -> None:
        btn = self._buttons.get(key)
        if btn is not None:
            btn.setEnabled(enabled)

    def current_key(self) -> str | None:
        for key, btn in self._buttons.items():
            if btn.isChecked():
                return key
        return None
