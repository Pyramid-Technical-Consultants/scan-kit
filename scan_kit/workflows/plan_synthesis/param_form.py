"""Dynamic Qt form builder for plan template parameters."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .energies import STANDARD_ENERGIES_MEV
from .params import ParamSpec

ReadParamsFn = Callable[[], dict[str, Any]]

_ENERGY_LIST_VISIBLE_ROWS = 8


def _list_height_for_rows(list_widget: QListWidget, rows: int) -> int:
    """Pixel height to show ``rows`` list entries (scroll for the rest)."""
    if list_widget.count() == 0:
        return 0
    row_h = list_widget.sizeHintForRow(0)
    if row_h <= 0:
        row_h = list_widget.fontMetrics().height() + 4
    return row_h * rows + 2 * list_widget.frameWidth()


class ParamFormWidget(QWidget):
    """Scrollable form built from :class:`ParamSpec` entries."""

    def __init__(
        self,
        specs: list[ParamSpec],
        values: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._specs = specs
        self._editors: dict[str, QWidget] = {}
        self._energy_list: QListWidget | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        initial = values or {}
        i = 0
        while i < len(specs):
            spec = specs[i]
            if spec.row_partner:
                i += 1
                continue

            partner = specs[i + 1] if i + 1 < len(specs) else None
            if partner is not None and partner.row_partner == spec.key:
                left_w = self._make_editor(spec, initial.get(spec.key, spec.default))
                right_w = self._make_editor(
                    partner, initial.get(partner.key, partner.default)
                )
                self._editors[spec.key] = left_w
                self._editors[partner.key] = right_w
                form.addRow(spec.label, self._make_inline_pair(spec, partner, left_w, right_w))
                i += 2
                continue

            widget = self._make_editor(spec, initial.get(spec.key, spec.default))
            self._editors[spec.key] = widget
            if spec.kind == "energy_multiselect":
                form.addRow(spec.label, widget)
            else:
                form.addRow(self._field_label(spec), widget)
            i += 1

        outer.addLayout(form)

    @staticmethod
    def _field_label(spec: ParamSpec) -> str:
        if spec.suffix:
            return f"{spec.label} ({spec.suffix})"
        return spec.label

    def _make_inline_pair(
        self,
        left_spec: ParamSpec,
        right_spec: ParamSpec,
        left_widget: QWidget,
        right_widget: QWidget,
    ) -> QWidget:
        row = QHBoxLayout()
        row.setSpacing(16)
        for spec, widget in ((left_spec, left_widget), (right_spec, right_widget)):
            group = QHBoxLayout()
            group.setSpacing(6)
            if spec.sub_label:
                sub = QLabel(f"{spec.sub_label}:")
                sub.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
                group.addWidget(sub)
            group.addWidget(widget, stretch=1)
            if spec.suffix:
                unit = QLabel(spec.suffix)
                unit.setStyleSheet("color: palette(mid);")
                unit.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
                group.addWidget(unit)
            row.addLayout(group, stretch=1)
        row.addStretch(1)
        box = QWidget()
        box.setLayout(row)
        box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return box

    def _make_editor(self, spec: ParamSpec, value: Any) -> QWidget:
        if spec.kind == "float":
            spin = QDoubleSpinBox()
            spin.setDecimals(spec.decimals)
            if spec.minimum is not None:
                spin.setMinimum(float(spec.minimum))
            if spec.maximum is not None:
                spin.setMaximum(float(spec.maximum))
            if spec.step is not None:
                spin.setSingleStep(float(spec.step))
            spin.setValue(float(value))
            return spin

        if spec.kind == "int":
            spin = QSpinBox()
            if spec.minimum is not None:
                spin.setMinimum(int(spec.minimum))
            if spec.maximum is not None:
                spin.setMaximum(int(spec.maximum))
            if spec.step is not None:
                spin.setSingleStep(int(spec.step))
            spin.setValue(int(value))
            return spin

        if spec.kind == "bool":
            box = QCheckBox()
            box.setChecked(bool(value))
            return box

        if spec.kind == "energy_multiselect":
            return self._make_energy_picker(value)

        raise ValueError(f"Unsupported param kind: {spec.kind}")

    def _make_energy_picker(self, value: Any) -> QWidget:
        selected = {float(e) for e in value} if isinstance(value, list) else set()

        box = QWidget()
        box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)

        btn_row = QHBoxLayout()
        select_all = QPushButton("Select all")
        clear_all = QPushButton("Clear all")
        btn_row.addWidget(select_all)
        btn_row.addWidget(clear_all)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        energy_list = QListWidget()
        energy_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for energy in reversed(STANDARD_ENERGIES_MEV):
            item = QListWidgetItem(f"{energy:g} MeV")
            item.setData(Qt.ItemDataRole.UserRole, float(energy))
            energy_list.addItem(item)
            if float(energy) in selected:
                item.setSelected(True)
        energy_list.setFixedHeight(
            _list_height_for_rows(energy_list, _ENERGY_LIST_VISIBLE_ROWS)
        )
        energy_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay.addWidget(energy_list)
        box.setFixedHeight(box.sizeHint().height())

        select_all.clicked.connect(energy_list.selectAll)
        clear_all.clicked.connect(energy_list.clearSelection)

        self._energy_list = energy_list
        return box

    def read_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for spec in self._specs:
            widget = self._editors[spec.key]
            if spec.kind == "float":
                out[spec.key] = float(widget.value())  # type: ignore[attr-defined]
            elif spec.kind == "int":
                out[spec.key] = int(widget.value())  # type: ignore[attr-defined]
            elif spec.kind == "bool":
                out[spec.key] = bool(widget.isChecked())  # type: ignore[attr-defined]
            elif spec.kind == "energy_multiselect":
                out[spec.key] = self._selected_energies()
        return out

    def _selected_energies(self) -> list[float]:
        if self._energy_list is None:
            return []
        energies: list[float] = []
        for i in range(self._energy_list.count()):
            item = self._energy_list.item(i)
            if item is not None and item.isSelected():
                val = item.data(Qt.ItemDataRole.UserRole)
                if val is not None:
                    energies.append(float(val))
        return energies


def build_param_form(
    specs: list[ParamSpec],
    values: dict[str, Any] | None = None,
    *,
    parent: QWidget | None = None,
) -> ParamFormWidget:
    return ParamFormWidget(specs, values, parent=parent)
