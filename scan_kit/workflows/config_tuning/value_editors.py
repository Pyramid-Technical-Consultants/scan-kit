"""Value type inference and Qt editor widgets for XML fields."""

from __future__ import annotations

import math
from enum import Enum

from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QWidget,
)


class ValueKind(str, Enum):
    STRING = "string"
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"


_BOOL_TAGS = frozenset(
    {
        "save_datafiles",
        "reverse_strips",
        "hidden",
        "draw",
        "fast_axis",
    }
)
_BOOL_ATTRS = frozenset({"hidden", "draw", "reverse_strips", "fast_axis"})


class _NoWheelSpinBox(QSpinBox):
    """Spin box that ignores the mouse wheel so scrolling passes to the form."""

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()


class _NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Double spin box that ignores the mouse wheel so scrolling passes to the form."""

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()

    def textFromValue(self, value: float) -> str:  # noqa: N802
        return format_float_display(value, max_decimals=self.decimals())


def _float_decimals_for_raw(raw: str) -> int:
    """Choose a sensible precision for *raw* without excess trailing zeros."""
    text = (raw or "").strip()
    if not text:
        return 6
    lower = text.lower()
    if "e" in lower:
        return 12
    if "." in text:
        fractional = text.split(".", 1)[1]
        return min(12, max(1, len(fractional.rstrip("0")) or len(fractional)))
    return 0


def format_float_display(value: float, *, max_decimals: int = 12) -> str:
    """Format a float for display without excessive trailing zeros."""
    number = float(value)
    if not math.isfinite(number):
        return str(number)

    abs_number = abs(number)
    if abs_number != 0.0 and (abs_number >= 1e6 or abs_number < 1e-4):
        return repr(number).replace("e+", "e")

    rounded = round(number, max_decimals)
    if float(int(rounded)) == rounded and abs(rounded) < 1e15:
        return str(int(rounded))

    text = f"{rounded:.{max_decimals}f}".rstrip("0").rstrip(".")
    return text or "0"


def infer_value_kind(raw: str, *, tag: str = "", attr: str = "") -> ValueKind:
    """Infer an editor kind from a serialized XML value."""
    text = (raw or "").strip()
    if not text:
        return ValueKind.STRING

    lower = text.lower()
    if lower in ("true", "false"):
        return ValueKind.BOOL
    if (tag in _BOOL_TAGS or attr in _BOOL_ATTRS) and text in ("0", "1"):
        return ValueKind.BOOL

    try:
        if "." not in text.lower() and "e" not in text.lower():
            int(text)
            return ValueKind.INT
    except ValueError:
        pass

    try:
        float(text)
        return ValueKind.FLOAT
    except ValueError:
        return ValueKind.STRING


def format_value(kind: ValueKind, value) -> str:
    """Serialize a Python value back to an XML string."""
    if kind is ValueKind.BOOL:
        if isinstance(value, str):
            return "1" if value.strip().lower() in ("1", "true", "yes", "on") else "0"
        return "1" if bool(value) else "0"
    if kind is ValueKind.INT:
        return str(int(value))
    if kind is ValueKind.FLOAT:
        return format_float_display(float(value))
    return str(value)


def read_widget_value(widget: QWidget, kind: ValueKind) -> str:
    """Read a widget and return the serialized XML string."""
    if isinstance(widget, QCheckBox):
        return format_value(ValueKind.BOOL, widget.isChecked())
    if isinstance(widget, (_NoWheelSpinBox, QSpinBox)):
        return format_value(ValueKind.INT, widget.value())
    if isinstance(widget, (_NoWheelDoubleSpinBox, QDoubleSpinBox)):
        return format_value(ValueKind.FLOAT, widget.value())
    if isinstance(widget, QLineEdit):
        return widget.text()
    return ""


def editor_minimum_width(kind: ValueKind, raw: str) -> int:
    """Minimum editor width so values (especially scientific notation) stay readable."""
    text = (raw or "").strip()
    if kind is ValueKind.BOOL:
        return 28
    if kind is ValueKind.STRING:
        return max(96, min(240, len(text) * 8 + 32))
    display_len = max(len(text), 6)
    if "e" in text.lower():
        display_len = max(display_len, 10)
    return min(220, max(104, display_len * 8 + 32))


def _apply_editor_width(editor: QWidget, kind: ValueKind, raw: str) -> None:
    width = editor_minimum_width(kind, raw)
    editor.setMinimumWidth(width)
    editor.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


def make_value_editor(kind: ValueKind, raw: str, *, tag: str = "", attr: str = "") -> QWidget:
    """Create a Qt editor widget for *raw* using *kind*."""
    text = raw or ""
    if kind is ValueKind.BOOL:
        box = QCheckBox()
        box.setChecked(text.strip().lower() in ("1", "true", "yes", "on"))
        _apply_editor_width(box, kind, text)
        return box

    if kind is ValueKind.INT:
        spin = _NoWheelSpinBox()
        spin.setRange(-2_000_000_000, 2_000_000_000)
        try:
            spin.setValue(int(text.strip()))
        except ValueError:
            spin.setValue(0)
        _apply_editor_width(spin, kind, text)
        return spin

    if kind is ValueKind.FLOAT:
        spin = _NoWheelDoubleSpinBox()
        spin.setDecimals(_float_decimals_for_raw(text))
        spin.setRange(-1e18, 1e18)
        try:
            spin.setValue(float(text.strip()))
        except ValueError:
            spin.setValue(0.0)
        _apply_editor_width(spin, kind, text)
        return spin

    edit = QLineEdit(text)
    edit.setClearButtonEnabled(True)
    _apply_editor_width(edit, kind, text)
    return edit


def set_widget_value(widget: QWidget, kind: ValueKind, raw: str) -> None:
    """Set a widget's display from a serialized XML string."""
    text = raw or ""
    if isinstance(widget, QCheckBox):
        widget.setChecked(text.strip().lower() in ("1", "true", "yes", "on"))
        return
    if isinstance(widget, (_NoWheelSpinBox, QSpinBox)):
        try:
            widget.setValue(int(text.strip()))
        except ValueError:
            widget.setValue(0)
        return
    if isinstance(widget, (_NoWheelDoubleSpinBox, QDoubleSpinBox)):
        try:
            widget.setValue(float(text.strip()))
        except ValueError:
            widget.setValue(0.0)
        return
    if isinstance(widget, QLineEdit):
        widget.setText(text)
        return
