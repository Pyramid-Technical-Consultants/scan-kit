"""Auto-generated Qt form for editing an XML DOM."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .collapsible_group import (
    CollapsibleGroupBox,
    PlainFieldset,
    make_form_section,
    make_plain_fieldset,
)

from .flow_layout import FlowWidget
from .labels import humanize_xml_label
from .value_editors import (
    ValueKind,
    infer_value_kind,
    make_value_editor,
    read_widget_value,
)
from .map2map_attr_registry import (
    filter_attribute_names,
    should_hide_map2map_child,
)
from .xml_bindings import (
    AttrBinding,
    BindingSet,
    TableBinding,
    TextBinding,
    attribute_names,
    is_homogeneous_attribute_row_group,
    table_columns_for_elements,
    use_attribute_table,
)

OnChangeFn = Callable[[], None]

_ATTRIBUTE_TABLE_MAX_VISIBLE_ROWS = 6
# Leaf elements with up to this many attributes use wrapping inline chips (e.g. precision).
_INLINE_ATTRIBUTE_FIELD_THRESHOLD = 2
_SCALAR_LABEL_WIDTH = 240


def _attribute_table_height(table: QTableWidget, row_count: int) -> int:
    """Height for an attribute table showing up to six rows without extra slack."""
    visible_rows = min(row_count, _ATTRIBUTE_TABLE_MAX_VISIBLE_ROWS)
    header_height = table.horizontalHeader().height()
    if header_height <= 0:
        header_height = table.horizontalHeader().sizeHint().height()
    if visible_rows == 0:
        return header_height + table.frameWidth() * 2

    if row_count > 0:
        table.resizeRowsToContents()
        row_height = max(table.rowHeight(0), table.verticalHeader().defaultSectionSize())
    else:
        row_height = table.verticalHeader().defaultSectionSize()

    return header_height + row_height * visible_rows + table.frameWidth() * 2


class XmlFormWidget(QScrollArea):
    """Scrollable form built recursively from an XML element tree."""

    def __init__(
        self,
        root: ET.Element,
        *,
        on_change: OnChangeFn | None = None,
        hide_unused_map2map: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_change = on_change
        self._hide_unused_map2map = hide_unused_map2map
        self._bindings = BindingSet()
        self._field_widgets: list[tuple[QWidget, ValueKind, str, str]] = []
        self._table_widgets: list[tuple[QTableWidget, TableBinding, list[ValueKind], list[str]]] = []

        host = QWidget()
        host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self._build_element(root, layout, title=root.tag, is_root=True)
        layout.addStretch(1)

        self.setWidget(host)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @property
    def bindings(self) -> BindingSet:
        return self._bindings

    def _visible_attribute_names(self, element: ET.Element) -> list[str]:
        names = attribute_names(element)
        if not self._hide_unused_map2map:
            return names
        return filter_attribute_names(element, names)

    def apply_to_dom(self) -> None:
        """Push current widget values into the bound XML DOM."""
        field_values = [
            read_widget_value(widget, kind)
            for widget, kind, _tag, _attr in self._field_widgets
        ]
        self._bindings.apply_field_values(field_values)

        for table, binding, _kinds, columns in self._table_widgets:
            rows: list[dict[str, str]] = []
            for row_idx in range(table.rowCount()):
                row_values: dict[str, str] = {}
                for col_idx, column in enumerate(columns):
                    item = table.item(row_idx, col_idx)
                    raw = item.text() if item is not None else binding.read_cell(row_idx, column)
                    row_values[column] = raw
                rows.append(row_values)
            for row_idx, row_values in enumerate(rows):
                binding.write_row(row_idx, row_values)

    def _emit_change(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def _connect_change(self, widget: QWidget) -> None:
        if hasattr(widget, "textChanged"):
            widget.textChanged.connect(self._emit_change)  # type: ignore[attr-defined]
        elif hasattr(widget, "valueChanged"):
            widget.valueChanged.connect(self._emit_change)  # type: ignore[attr-defined]
        elif hasattr(widget, "stateChanged"):
            widget.stateChanged.connect(self._emit_change)  # type: ignore[attr-defined]

    def _register_field(
        self,
        element: ET.Element,
        editor: QWidget,
        kind: ValueKind,
        *,
        attr: str | None = None,
    ) -> None:
        if attr:
            binding: TextBinding | AttrBinding = AttrBinding(element, attr)
        else:
            binding = TextBinding(element)
        self._bindings.fields.append(binding)
        self._field_widgets.append((editor, kind, element.tag, attr or ""))

    def _make_field_chip(
        self,
        label: str,
        element: ET.Element,
        *,
        attr: str | None = None,
    ) -> QWidget:
        raw = element.get(attr, "") if attr else (element.text or "")
        kind = infer_value_kind(raw, tag=element.tag, attr=attr or "")
        editor = make_value_editor(kind, raw, tag=element.tag, attr=attr or "")
        self._connect_change(editor)
        self._register_field(element, editor, kind, attr=attr)

        chip = QWidget()
        row = QHBoxLayout(chip)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        label_widget = QLabel(humanize_xml_label(label))
        label_widget.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(label_widget)
        row.addWidget(editor)

        spacing = row.spacing()
        margins = row.contentsMargins()
        editor_width = max(editor.minimumWidth(), editor.sizeHint().width())
        chip.setMinimumWidth(
            label_widget.sizeHint().width()
            + editor_width
            + spacing
            + margins.left()
            + margins.right()
        )
        chip.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        return chip

    def _make_attributes_flow_widget(
        self,
        element: ET.Element,
        *,
        include_value: bool = False,
    ) -> QWidget | None:
        if not element.attrib and not (include_value and (element.text or "").strip()):
            return None

        host = FlowWidget(h_spacing=8, v_spacing=6)
        for attr in self._visible_attribute_names(element):
            host.addWidget(self._make_field_chip(attr, element, attr=attr))
        if include_value and (element.text or "").strip():
            host.addWidget(self._make_field_chip("value", element))
        return host

    def _add_attributes_flow(
        self,
        parent_layout: QVBoxLayout,
        element: ET.Element,
        *,
        include_value: bool = False,
    ) -> None:
        host = self._make_attributes_flow_widget(element, include_value=include_value)
        if host is not None:
            parent_layout.addWidget(host)

    def _is_value_only_scalar(self, element: ET.Element) -> bool:
        """True when *element* has no attributes or child elements, only text."""
        return not element.attrib and not list(element)

    def _wrap_labeled_editor_row(self, label: str, editor: QWidget) -> QWidget:
        """Left-aligned label + editor row for stacked scalar fields."""
        row_widget = QWidget()
        row_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        label_widget = QLabel(humanize_xml_label(label))
        label_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label_widget.setFixedWidth(_SCALAR_LABEL_WIDTH)
        row.addWidget(label_widget)
        row.addWidget(editor)
        row.addStretch(1)
        return row_widget

    def _build_attribute_row(self, element: ET.Element, attr: str) -> QWidget:
        """Label + editor row for one attribute on *element*."""
        raw = element.get(attr, "")
        kind = infer_value_kind(raw, tag=element.tag, attr=attr)
        editor = make_value_editor(kind, raw, tag=element.tag, attr=attr)
        self._connect_change(editor)
        self._register_field(element, editor, kind, attr=attr)
        return self._wrap_labeled_editor_row(attr, editor)

    def _attribute_edit_count(self, element: ET.Element, *, include_value: bool = False) -> int:
        count = len(element.attrib)
        if include_value and (element.text or "").strip():
            count += 1
        return count

    def _uses_inline_attribute_flow(
        self,
        element: ET.Element,
        *,
        include_value: bool = False,
    ) -> bool:
        return (
            self._attribute_edit_count(element, include_value=include_value)
            >= _INLINE_ATTRIBUTE_FIELD_THRESHOLD
        )

    def _add_attribute_fieldset_content(
        self,
        layout: QVBoxLayout,
        element: ET.Element,
        *,
        include_value: bool = False,
    ) -> None:
        if self._uses_inline_attribute_flow(element, include_value=include_value):
            flow = self._make_attributes_flow_widget(
                element,
                include_value=include_value,
            )
            if flow is not None:
                layout.addWidget(flow)
            return

        for attr in self._visible_attribute_names(element):
            layout.addWidget(self._build_attribute_row(element, attr))
        if include_value and (element.text or "").strip():
            layout.addWidget(self._build_value_row(element, "value"))

    def _build_attribute_fieldset(
        self,
        element: ET.Element,
        title: str,
        *,
        include_value: bool = False,
    ) -> PlainFieldset:
        box, layout = make_plain_fieldset(
            humanize_xml_label(title),
            margins=(8, 8, 8, 8),
            spacing=6,
        )
        self._add_attribute_fieldset_content(
            layout,
            element,
            include_value=include_value,
        )
        return box

    def _build_value_row(self, element: ET.Element, label: str) -> QWidget:
        """Compact label + editor row for a text-only scalar element."""
        raw = element.text or ""
        kind = infer_value_kind(raw, tag=element.tag)
        editor = make_value_editor(kind, raw, tag=element.tag)
        self._connect_change(editor)
        self._register_field(element, editor, kind)
        return self._wrap_labeled_editor_row(label, editor)

    def _build_scalar_widget(self, element: ET.Element, title: str) -> QWidget:
        include_value = bool((element.text or "").strip())
        if self._is_value_only_scalar(element):
            return self._build_value_row(element, element.tag)

        if self._attribute_edit_count(element, include_value=include_value) >= 2:
            fieldset_title = self._inline_row_label(element, element.tag, 0)
            return self._build_attribute_fieldset(
                element,
                fieldset_title,
                include_value=include_value,
            )

        if len(element.attrib) == 1 and not include_value:
            attr = next(iter(element.attrib))
            return self._build_attribute_row(element, attr)

        widget = self._make_attributes_flow_widget(element, include_value=include_value)
        if widget is not None:
            return widget
        return self._build_value_row(element, title)

    def _make_attribute_table(self, elements: list[ET.Element]) -> QTableWidget:
        columns = table_columns_for_elements(elements)
        if self._hide_unused_map2map and elements:
            columns = filter_attribute_names(elements[0], columns)
        binding = TableBinding(elements=list(elements), columns=columns)
        self._bindings.tables.append(binding)

        kinds = [
            infer_value_kind(
                elements[0].get(col, "") if elements else "",
                tag=elements[0].tag if elements else "",
                attr=col,
            )
            for col in columns
        ]

        table = QTableWidget(len(elements), len(columns))
        table.setHorizontalHeaderLabels([humanize_xml_label(col) for col in columns])
        header = table.horizontalHeader()
        header.setMinimumSectionSize(48)
        header.setDefaultSectionSize(96)
        header.setStretchLastSection(True)
        for col_idx in range(len(columns)):
            header.setSectionResizeMode(col_idx, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setMinimumWidth(0)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        for row_idx, element in enumerate(elements):
            for col_idx, column in enumerate(columns):
                raw = element.get(column, "")
                item = QTableWidgetItem(raw)
                table.setItem(row_idx, col_idx, item)

        row_count = len(elements)
        table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
            if row_count > _ATTRIBUTE_TABLE_MAX_VISIBLE_ROWS
            else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        table.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        table.setFixedHeight(_attribute_table_height(table, row_count))

        table.itemChanged.connect(lambda *_: self._emit_change())
        self._table_widgets.append((table, binding, kinds, columns))
        return table

    def _build_table(self, elements: list[ET.Element], title: str) -> PlainFieldset:
        box, layout = make_plain_fieldset(
            humanize_xml_label(f"{title} ({len(elements)} rows)")
        )
        layout.addWidget(self._make_attribute_table(elements))
        return box

    @staticmethod
    @staticmethod
    def _group_children_raw(element: ET.Element) -> list[tuple[str, list[ET.Element]]]:
        children = list(element)
        groups: list[tuple[str, list[ET.Element]]] = []
        idx = 0
        while idx < len(children):
            child = children[idx]
            run = [child]
            j = idx + 1
            while j < len(children) and children[j].tag == child.tag:
                run.append(children[j])
                j += 1
            groups.append((child.tag, run))
            idx = j
        return groups

    def _group_children(self, element: ET.Element) -> list[tuple[str, list[ET.Element]]]:
        if not self._hide_unused_map2map:
            return self._group_children_raw(element)
        filtered = ET.Element(element.tag, element.attrib)
        for child in element:
            if not should_hide_map2map_child(element, child.tag):
                filtered.append(child)
        return self._group_children_raw(filtered)

    @staticmethod
    def _sole_fieldset_group(
        child_groups: list[tuple[str, list[ET.Element]]],
    ) -> tuple[str, list[ET.Element], str] | None:
        """Return ``(tag, elements, kind)`` when *child_groups* is one fieldset child.

        *kind* is ``"table"`` for homogeneous attribute tables or ``"container"`` for a
        lone nested element with children.
        """
        if len(child_groups) != 1:
            return None
        tag, group = child_groups[0]
        if use_attribute_table(group):
            return tag, group, "table"
        if len(group) == 1 and list(group[0]):
            return tag, group, "container"
        return None

    def _element_has_own_attrs(self, element: ET.Element, *, is_root: bool) -> bool:
        return bool(element.attrib) or (
            is_root and bool((element.text or "").strip())
        )

    def _device_name_from_element(self, element: ET.Element) -> str | None:
        """Return the ``device/@name`` marker when *element* uses one."""
        device = element.find("device")
        if device is None or list(device) or (device.text or "").strip():
            return None
        name = device.get("name")
        return name or None

    @staticmethod
    def _inline_row_label(element: ET.Element, tag: str, index: int) -> str:
        for key in ("name", "in_units", "units", "min_energy", "out_units"):
            value = element.get(key)
            if value:
                return f"{tag} ({value})"
        return f"{tag} [{index + 1}]"

    def _sibling_element_title(
        self,
        element: ET.Element,
        tag: str,
        index: int,
        group_len: int,
    ) -> str:
        device_name = self._device_name_from_element(element)
        if device_name:
            return device_name
        if group_len == 1:
            return tag
        return f"{tag} [{index + 1}]"

    def _is_simple_scalar_element(self, element: ET.Element) -> bool:
        """True for leaf scalars: text values, ``device/@name``, and small attribute combos."""
        if list(element):
            return False
        attr_count = len(element.attrib)
        has_text = bool((element.text or "").strip())
        if attr_count == 0:
            return has_text
        if has_text:
            return False
        return attr_count <= _INLINE_ATTRIBUTE_FIELD_THRESHOLD

    def _is_simple_scalar_group(self, _tag: str, group: list[ET.Element]) -> bool:
        return len(group) == 1 and self._is_simple_scalar_element(group[0])

    def _make_labeled_attribute_flow_chip(self, tag: str, element: ET.Element) -> QWidget:
        """Inline attribute chips prefixed with a short group label (e.g. precision + units/value)."""
        chip = QWidget()
        chip.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(chip)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        title = QLabel(humanize_xml_label(self._inline_row_label(element, tag, 0)))
        title.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(title)
        content_width = title.sizeHint().width() + row.spacing()
        for attr in self._visible_attribute_names(element):
            field_chip = self._make_field_chip(attr, element, attr=attr)
            row.addWidget(field_chip)
            content_width += field_chip.minimumWidth() + row.spacing()
        margins = row.contentsMargins()
        chip.setMinimumWidth(content_width + margins.left() + margins.right())
        return chip

    def _make_simple_scalars_flow(
        self,
        items: list[tuple[str, ET.Element]],
    ) -> FlowWidget:
        host = FlowWidget(h_spacing=8, v_spacing=6)
        for tag, element in items:
            attr_count = len(element.attrib)
            has_text = bool((element.text or "").strip())
            if attr_count >= 2 and not has_text:
                host.addWidget(self._make_labeled_attribute_flow_chip(tag, element))
            elif attr_count == 1 and not has_text:
                attr = next(iter(self._visible_attribute_names(element)))
                host.addWidget(self._make_field_chip(attr, element, attr=attr))
            else:
                host.addWidget(self._make_field_chip(tag, element))
        return host

    @staticmethod
    def _should_flatten_container(element: ET.Element) -> bool:
        """True for attribute-less wrappers whose children should render directly."""
        if element.attrib or (element.text or "").strip():
            return False
        if not list(element):
            return False
        return len(XmlFormWidget._group_children_raw(element)) != 1

    @staticmethod
    def _child_groups_use_nested_sections(
        child_groups: list[tuple[str, list[ET.Element]]],
    ) -> bool:
        """True when child groups render nested section containers (not leaf widgets)."""
        sole_at_parent = len(child_groups) == 1
        for _tag, group in child_groups:
            if XmlFormWidget._group_produces_nested_sections(
                group,
                sole_at_parent=sole_at_parent,
            ):
                return True
        return False

    @staticmethod
    def _group_produces_nested_sections(
        group: list[ET.Element],
        *,
        sole_at_parent: bool = False,
    ) -> bool:
        if len(group) == 1 and not list(group[0]):
            return False
        if use_attribute_table(group):
            return False
        if is_homogeneous_attribute_row_group(group):
            return False
        if len(group) == 1 and list(group[0]):
            inner = group[0]
            if sole_at_parent and XmlFormWidget._should_flatten_container(inner):
                return XmlFormWidget._child_groups_use_nested_sections(
                    XmlFormWidget._group_children_raw(inner)
                )
            return XmlFormWidget._child_groups_use_nested_sections(
                XmlFormWidget._group_children_raw(inner)
            )
        return True

    def _sections_added_by_build_element(
        self,
        element: ET.Element,
        *,
        title: str,
    ) -> int:
        """Count section widgets one ``_build_element`` call adds to its parent."""
        children = list(element)
        if not children:
            return 0

        child_groups = self._group_children(element)
        sole = self._sole_fieldset_group(child_groups)
        has_own_attrs = self._element_has_own_attrs(element, is_root=False)

        if sole is not None and not has_own_attrs:
            _tag, group, kind = sole
            if kind == "container":
                return self._sections_added_by_build_element(
                    group[0],
                    title=f"{title} / {_tag}",
                )
            return 1

        if sole is not None and has_own_attrs:
            return 1

        return 1

    def _scalar_widget_is_section(self, element: ET.Element) -> bool:
        """True when ``_build_scalar_widget`` renders a titled fieldset section."""
        if self._is_value_only_scalar(element):
            return False
        include_value = bool((element.text or "").strip())
        return self._attribute_edit_count(element, include_value=include_value) >= 2

    def _child_groups_section_count(
        self,
        child_groups: list[tuple[str, list[ET.Element]]],
    ) -> int:
        """Count plain or collapsible fieldsets rendered as direct child groups."""
        total = 0
        idx = 0
        while idx < len(child_groups):
            tag, group = child_groups[idx]
            if self._is_simple_scalar_group(tag, group):
                while idx < len(child_groups):
                    batch_tag, batch_group = child_groups[idx]
                    if not self._is_simple_scalar_group(batch_tag, batch_group):
                        break
                    idx += 1
                continue

            if use_attribute_table(group):
                total += 1
                idx += 1
                continue
            if is_homogeneous_attribute_row_group(group):
                if all(self._is_simple_scalar_element(child) for child in group):
                    idx += 1
                    continue
                total += len(group)
                idx += 1
                continue
            if len(group) == 1 and not list(group[0]):
                if self._scalar_widget_is_section(group[0]):
                    total += 1
                idx += 1
                continue
            if len(group) == 1 and list(group[0]):
                child = group[0]
                if len(child_groups) == 1 and self._should_flatten_container(child):
                    total += self._child_groups_section_count(
                        self._group_children(child)
                    )
                else:
                    total += self._sections_added_by_build_element(child, title=tag)
                idx += 1
                continue
            for child in group:
                total += self._sections_added_by_build_element(child, title=tag)
            idx += 1
        return total

    def _child_groups_has_collapsible_child(
        self,
        child_groups: list[tuple[str, list[ET.Element]]],
    ) -> bool:
        """True when a child group renders a collapsible section."""
        idx = 0
        while idx < len(child_groups):
            tag, group = child_groups[idx]
            if self._is_simple_scalar_group(tag, group):
                while idx < len(child_groups):
                    batch_tag, batch_group = child_groups[idx]
                    if not self._is_simple_scalar_group(batch_tag, batch_group):
                        break
                    idx += 1
                continue

            if use_attribute_table(group):
                idx += 1
                continue
            if is_homogeneous_attribute_row_group(group):
                idx += 1
                continue
            if len(group) == 1 and not list(group[0]):
                idx += 1
                continue
            if len(group) == 1 and list(group[0]):
                child = group[0]
                if len(child_groups) == 1 and self._should_flatten_container(child):
                    if self._child_groups_has_collapsible_child(
                        self._group_children(child)
                    ):
                        return True
                elif self._element_needs_collapsible_section(child, is_root=False):
                    return True
                idx += 1
                continue
            for child in group:
                if self._element_needs_collapsible_section(child, is_root=False):
                    return True
            idx += 1
        return False

    def _element_needs_collapsible_section(
        self,
        element: ET.Element,
        *,
        is_root: bool = False,
    ) -> bool:
        """True when *element* should use a collapsible (not plain) section wrapper."""
        children = list(element)
        if not children:
            return False

        child_groups = self._group_children(element)
        sole = self._sole_fieldset_group(child_groups)
        has_own_attrs = self._element_has_own_attrs(element, is_root=is_root)

        if sole is not None and not has_own_attrs:
            _tag, group, kind = sole
            if kind == "container":
                return self._element_needs_collapsible_section(group[0], is_root=False)
            return False

        if sole is not None and has_own_attrs:
            _tag, group, kind = sole
            if kind == "table":
                return False
            return self._element_needs_collapsible_section(group[0], is_root=False)

        if self._child_groups_section_count(child_groups) >= 1:
            return True
        if self._child_groups_has_collapsible_child(child_groups):
            return True
        return self._child_groups_use_nested_sections(child_groups)

    def _add_child_groups(self, layout: QVBoxLayout, child_groups: list[tuple[str, list[ET.Element]]]) -> None:
        idx = 0
        while idx < len(child_groups):
            tag, group = child_groups[idx]
            if self._is_simple_scalar_group(tag, group):
                batch: list[tuple[str, ET.Element]] = []
                while idx < len(child_groups):
                    batch_tag, batch_group = child_groups[idx]
                    if not self._is_simple_scalar_group(batch_tag, batch_group):
                        break
                    batch.append((batch_tag, batch_group[0]))
                    idx += 1
                layout.addWidget(self._make_simple_scalars_flow(batch))
                continue

            if use_attribute_table(group):
                layout.addWidget(self._build_table(group, tag))
                idx += 1
                continue
            if is_homogeneous_attribute_row_group(group):
                if all(self._is_simple_scalar_element(child) for child in group):
                    layout.addWidget(
                        self._make_simple_scalars_flow([(tag, child) for child in group])
                    )
                else:
                    for index, child in enumerate(group):
                        row_title = self._inline_row_label(child, tag, index)
                        layout.addWidget(self._build_attribute_fieldset(child, row_title))
                idx += 1
                continue
            if len(group) == 1 and not list(group[0]):
                layout.addWidget(self._build_scalar_widget(group[0], tag))
                idx += 1
                continue
            if len(group) == 1 and list(group[0]):
                child = group[0]
                child_title = self._sibling_element_title(child, tag, 0, 1)
                if len(child_groups) == 1 and self._should_flatten_container(child):
                    self._add_child_groups(layout, self._group_children(child))
                else:
                    self._build_element(child, layout, title=child_title)
                idx += 1
                continue
            for i, child in enumerate(group):
                child_title = self._sibling_element_title(child, tag, i, len(group))
                self._build_element(child, layout, title=child_title)
            idx += 1

    def _build_element(
        self,
        element: ET.Element,
        parent_layout: QVBoxLayout,
        *,
        title: str,
        is_root: bool = False,
    ) -> None:
        children = list(element)
        if not children:
            parent_layout.addWidget(self._build_scalar_widget(element, title))
            return

        child_groups = self._group_children(element)
        sole = self._sole_fieldset_group(child_groups)
        has_own_attrs = self._element_has_own_attrs(element, is_root=is_root)

        if is_root and sole is not None:
            tag, group, kind = sole
            if has_own_attrs:
                self._add_attributes_flow(
                    parent_layout,
                    element,
                    include_value=bool((element.text or "").strip()),
                )
            if kind == "container":
                inner = group[0]
                if self._should_flatten_container(inner):
                    self._add_child_groups(
                        parent_layout,
                        self._group_children(inner),
                    )
                else:
                    self._build_element(inner, parent_layout, title=tag, is_root=False)
            else:
                parent_layout.addWidget(self._build_table(group, tag))
            return

        if sole is not None and not has_own_attrs:
            tag, group, kind = sole
            merged = f"{title} / {tag}"
            if kind == "container":
                inner = group[0]
                if self._should_flatten_container(inner):
                    self._add_child_groups(
                        parent_layout,
                        self._group_children(inner),
                    )
                else:
                    self._build_element(inner, parent_layout, title=merged)
                return
            parent_layout.addWidget(self._build_table(group, merged))
            return

        if sole is not None and has_own_attrs:
            tag, group, kind = sole
            merged = f"{title} / {tag}"
            collapsible = kind != "table" and self._element_needs_collapsible_section(
                group[0],
                is_root=False,
            )
            if kind == "table":
                box, layout = make_form_section(
                    humanize_xml_label(f"{merged} ({len(group)} rows)"),
                    collapsible=collapsible,
                )
            else:
                box, layout = make_form_section(
                    humanize_xml_label(merged),
                    collapsible=collapsible,
                )
            self._add_attributes_flow(
                layout,
                element,
                include_value=is_root and bool((element.text or "").strip()),
            )
            if kind == "table":
                layout.addWidget(self._make_attribute_table(group))
            else:
                self._build_element(group[0], layout, title=merged)
            parent_layout.addWidget(box)
            return

        box, layout = make_form_section(
            humanize_xml_label(title),
            collapsible=self._element_needs_collapsible_section(element, is_root=is_root),
        )

        if has_own_attrs:
            self._add_attributes_flow(
                layout,
                element,
                include_value=is_root and bool((element.text or "").strip()),
            )
        self._add_child_groups(layout, child_groups)

        parent_layout.addWidget(box)
