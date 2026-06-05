"""Dynamic Qt form builder for plan template parameters."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .energy_picker import EnergyPickerWidget
from .params import (
    PARAM_FIELD_SET_LABELS,
    PARAM_FIELD_SET_ORDER,
    ParamSpec,
    QuickSet,
)

ReadParamsFn = Callable[[], dict[str, Any]]


class _RadioButtonGroupField(QWidget):
    """Horizontal :class:`QRadioButton` group backed by :class:`QButtonGroup`."""

    def __init__(
        self,
        choices: tuple[tuple[Any, str], ...],
        value: Any,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._value_by_id: dict[int, Any] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        group = QButtonGroup(self)
        for index, (choice_value, choice_label) in enumerate(choices):
            radio = QRadioButton(choice_label)
            group.addButton(radio, index)
            self._value_by_id[index] = choice_value
            layout.addWidget(radio)
            if choice_value == value:
                radio.setChecked(True)

        if group.checkedId() < 0 and self._value_by_id:
            group.button(0).setChecked(True)

        self._group = group

    @property
    def button_group(self) -> QButtonGroup:
        return self._group

    def value(self) -> Any:
        checked_id = self._group.checkedId()
        if checked_id < 0:
            return next(iter(self._value_by_id.values()))
        return self._value_by_id[checked_id]


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
        self._energy_picker: EnergyPickerWidget | None = None
        self._form_rows: list[dict[str, Any]] = []
        self._field_set_boxes: dict[str, QGroupBox] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        label_width = self._label_column_width(specs)
        initial = values or {}
        grouped = self._group_specs(specs)
        for field_set in PARAM_FIELD_SET_ORDER:
            group_specs = grouped.get(field_set, [])
            if not group_specs:
                continue
            box = QGroupBox(PARAM_FIELD_SET_LABELS[field_set])
            box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            form = QFormLayout(box)
            form.setContentsMargins(8, 8, 8, 8)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
            form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            form.setVerticalSpacing(4)
            form.setHorizontalSpacing(0)
            self._populate_form(form, group_specs, initial, label_width, field_set)
            self._field_set_boxes[field_set] = box
            outer.addWidget(box)

        self._connect_visibility_watchers()
        self._refresh_visibility()

    @staticmethod
    def _group_specs(specs: list[ParamSpec]) -> dict[str, list[ParamSpec]]:
        grouped: dict[str, list[ParamSpec]] = {}
        for spec in specs:
            grouped.setdefault(spec.field_set, []).append(spec)
        return grouped

    def _populate_form(
        self,
        form: QFormLayout,
        specs: list[ParamSpec],
        initial: dict[str, Any],
        label_width: int,
        field_set: str,
    ) -> None:
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
                inline = self._make_inline_pair(spec, partner, left_w, right_w)
                field = self._wrap_field_with_quick_sets(inline, spec.quick_sets)
                self._add_form_row(
                    form,
                    spec.label,
                    field,
                    [spec, partner],
                    label_width=label_width,
                    field_set=field_set,
                )
                i += 2
                continue

            widget = self._make_editor(spec, initial.get(spec.key, spec.default))
            self._editors[spec.key] = widget
            if spec.kind == "energy_multiselect":
                form.addRow(widget)
                self._register_form_row([spec], None, widget, widget, field_set)
            else:
                self._add_form_row(
                    form,
                    self._field_label(spec),
                    widget,
                    [spec],
                    label_width=label_width,
                    field_set=field_set,
                )
            i += 1

    @staticmethod
    def _field_label(spec: ParamSpec) -> str:
        if spec.suffix:
            return f"{spec.label} ({spec.suffix})"
        return spec.label

    @staticmethod
    def _row_label(text: str) -> QLabel:
        """Left-column label that stays vertically centered in the form row."""
        label = QLabel(text)
        label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        label.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Preferred,
        )
        return label

    def _label_column_width(self, specs: list[ParamSpec]) -> int:
        texts: list[str] = []
        i = 0
        while i < len(specs):
            spec = specs[i]
            if spec.row_partner:
                i += 1
                continue
            partner = specs[i + 1] if i + 1 < len(specs) else None
            if partner is not None and partner.row_partner == spec.key:
                texts.append(spec.label)
                i += 2
                continue
            if spec.kind == "energy_multiselect":
                i += 1
                continue
            texts.append(self._field_label(spec))
            i += 1
        if not texts:
            return 0
        fm = self.fontMetrics()
        return max(fm.horizontalAdvance(text) for text in texts) + 8

    def _add_form_row(
        self,
        form: QFormLayout,
        label_text: str,
        field: QWidget,
        specs: list[ParamSpec],
        *,
        label_width: int,
        field_set: str,
    ) -> None:
        _vc = Qt.AlignmentFlag.AlignVCenter
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        label = self._row_label(label_text)
        label.setFixedWidth(label_width)
        layout.addWidget(label, alignment=_vc | Qt.AlignmentFlag.AlignRight)
        layout.addWidget(field, stretch=1, alignment=_vc | Qt.AlignmentFlag.AlignLeft)
        form.addRow(row)
        self._register_form_row(specs, label, field, row, field_set)

    def _register_form_row(
        self,
        specs: list[ParamSpec],
        label: QLabel | None,
        field: QWidget,
        row: QWidget,
        field_set: str,
    ) -> None:
        self._form_rows.append(
            {
                "specs": specs,
                "label": label,
                "field": field,
                "row": row,
                "field_set": field_set,
            }
        )

    def _connect_visibility_watchers(self) -> None:
        watch_keys = {
            key
            for spec in self._specs
            if spec.visible_when
            for key in spec.visible_when
        }
        for key in watch_keys:
            editor = self._editors.get(key)
            if editor is None:
                continue
            if isinstance(editor, QComboBox):
                editor.currentIndexChanged.connect(self._refresh_visibility)
            elif isinstance(editor, _RadioButtonGroupField):
                editor.button_group.idClicked.connect(self._refresh_visibility)

    def _refresh_visibility(self) -> None:
        current = self.read_params()
        field_set_visible = {field_set: False for field_set in self._field_set_boxes}

        for row in self._form_rows:
            visible = all(
                self._spec_visible(spec, current) for spec in row["specs"]
            )
            row["row"].setVisible(visible)
            if visible:
                field_set_visible[row["field_set"]] = True

        for field_set, box in self._field_set_boxes.items():
            box.setVisible(field_set_visible[field_set])

    @staticmethod
    def _spec_visible(spec: ParamSpec, current: dict[str, Any]) -> bool:
        if not spec.visible_when:
            return True
        for key, allowed in spec.visible_when.items():
            if current.get(key) not in allowed:
                return False
        return True

    def _wrap_field_with_quick_sets(
        self,
        field: QWidget,
        quick_sets: tuple[QuickSet, ...],
    ) -> QWidget:
        if not quick_sets:
            return field
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(field, stretch=1)
        layout.addWidget(self._make_quick_button_bar(quick_sets))
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return row

    @staticmethod
    def _make_compact_button(label: str) -> QPushButton:
        button = QPushButton(label)
        button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        button.setMinimumWidth(0)
        button.setStyleSheet("QPushButton { padding: 2px 6px; min-width: 0px; }")
        return button

    def _make_quick_button_bar(self, quick_sets: tuple[QuickSet, ...]) -> QWidget:
        _vc = Qt.AlignmentFlag.AlignVCenter
        box = QWidget()
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for quick_set in quick_sets:
            button = self._make_compact_button(quick_set.label)
            button.clicked.connect(
                lambda _checked=False, preset=quick_set: self._apply_quick_set(preset)
            )
            layout.addWidget(button, alignment=_vc)
        box.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        return box

    def _apply_quick_set(self, quick_set: QuickSet) -> None:
        for key, value in quick_set.values.items():
            editor = self._editors.get(key)
            if editor is None:
                continue
            if isinstance(editor, QDoubleSpinBox):
                editor.setValue(float(value))
            elif isinstance(editor, QSpinBox):
                editor.setValue(int(value))
            elif isinstance(editor, QCheckBox):
                editor.setChecked(bool(value))
            elif isinstance(editor, QComboBox):
                index = editor.findData(value)
                if index >= 0:
                    editor.setCurrentIndex(index)

    def _make_inline_pair(
        self,
        left_spec: ParamSpec,
        right_spec: ParamSpec,
        left_widget: QWidget,
        right_widget: QWidget,
    ) -> QWidget:
        _vc = Qt.AlignmentFlag.AlignVCenter
        row = QHBoxLayout()
        row.setSpacing(16)
        row.setAlignment(_vc)
        for spec, widget in ((left_spec, left_widget), (right_spec, right_widget)):
            group = QHBoxLayout()
            group.setSpacing(6)
            group.setAlignment(_vc)
            if spec.sub_label:
                sub = QLabel(f"{spec.sub_label}:")
                sub.setAlignment(_vc | Qt.AlignmentFlag.AlignRight)
                sub.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
                group.addWidget(sub, alignment=_vc)
            group.addWidget(widget, stretch=1, alignment=_vc)
            if spec.suffix:
                unit = QLabel(spec.suffix)
                unit.setStyleSheet("color: palette(mid);")
                unit.setAlignment(_vc | Qt.AlignmentFlag.AlignLeft)
                unit.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
                group.addWidget(unit, alignment=_vc)
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
            spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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
            spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return spin

        if spec.kind == "bool":
            box = QCheckBox()
            box.setChecked(bool(value))
            box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            return box

        if spec.kind == "choice":
            combo = QComboBox()
            for choice_value, choice_label in spec.choices:
                combo.addItem(choice_label, choice_value)
            index = combo.findData(value)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return combo

        if spec.kind == "button_group":
            field = _RadioButtonGroupField(spec.choices, value)
            field.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            return field

        if spec.kind == "energy_multiselect":
            return self._make_energy_picker(value)

        raise ValueError(f"Unsupported param kind: {spec.kind}")

    def _make_energy_picker(self, value: Any) -> EnergyPickerWidget:
        selected = value if isinstance(value, list) else None
        picker = EnergyPickerWidget(selected=selected)
        self._energy_picker = picker
        return picker

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
            elif spec.kind == "choice":
                out[spec.key] = widget.currentData()  # type: ignore[attr-defined]
            elif spec.kind == "button_group":
                out[spec.key] = widget.value()  # type: ignore[attr-defined]
            elif spec.kind == "energy_multiselect":
                out[spec.key] = self._selected_energies()
        return out

    def _selected_energies(self) -> list[float]:
        if self._energy_picker is None:
            return []
        return self._energy_picker.selected_energies()


def build_param_form(
    specs: list[ParamSpec],
    values: dict[str, Any] | None = None,
    *,
    parent: QWidget | None = None,
) -> ParamFormWidget:
    return ParamFormWidget(specs, values, parent=parent)
