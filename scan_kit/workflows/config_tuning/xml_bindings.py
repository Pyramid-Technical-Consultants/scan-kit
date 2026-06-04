"""Pure-Python bindings between XML DOM nodes and serialized field values."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Protocol


class FieldBinding(Protocol):
    def read(self) -> str: ...

    def write(self, value: str) -> None: ...


@dataclass
class TextBinding:
    element: ET.Element

    def read(self) -> str:
        return self.element.text or ""

    def write(self, value: str) -> None:
        self.element.text = value


@dataclass
class AttrBinding:
    element: ET.Element
    attr: str

    def read(self) -> str:
        return self.element.get(self.attr, "")

    def write(self, value: str) -> None:
        self.element.set(self.attr, value)


@dataclass
class TableBinding:
    """Homogeneous sibling elements edited as a table of attributes."""

    elements: list[ET.Element] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)

    def read_cell(self, row: int, column: str) -> str:
        return self.elements[row].get(column, "")

    def write_cell(self, row: int, column: str, value: str) -> None:
        self.elements[row].set(column, value)

    def read_row(self, row: int) -> dict[str, str]:
        return {col: self.read_cell(row, col) for col in self.columns}

    def write_row(self, row: int, values: dict[str, str]) -> None:
        for col in self.columns:
            if col in values:
                self.write_cell(row, col, values[col])


@dataclass
class BindingSet:
    fields: list[FieldBinding] = field(default_factory=list)
    tables: list[TableBinding] = field(default_factory=list)

    def apply_field_values(self, values: list[str]) -> None:
        if len(values) != len(self.fields):
            raise ValueError("field value count does not match bindings")
        for binding, value in zip(self.fields, values):
            binding.write(value)

    def apply_table_values(self, table_index: int, rows: list[dict[str, str]]) -> None:
        table = self.tables[table_index]
        if len(rows) != len(table.elements):
            raise ValueError("table row count does not match bindings")
        for row_idx, row_values in enumerate(rows):
            table.write_row(row_idx, row_values)


def use_attribute_table(children: list[ET.Element]) -> bool:
    """True when repeated attribute rows should render as a table."""
    return is_homogeneous_attribute_row_group(children) and len(children) > 2


def is_homogeneous_attribute_row_group(children: list[ET.Element]) -> bool:
    """True when siblings form an attribute-heavy repeated row group."""
    if len(children) < 2:
        return False
    tag = children[0].tag
    if any(child.tag != tag for child in children):
        return False
    if any(list(child) for child in children):
        return False
    if not all(not (child.text or "").strip() for child in children):
        return False
    return all(len(child.attrib) >= 1 for child in children)


def attribute_names(element: ET.Element) -> list[str]:
    """Return attribute names in document order."""
    return list(element.attrib)


def table_columns_for_elements(elements: list[ET.Element]) -> list[str]:
    """Union of attribute names across *elements*, preserving first-seen document order."""
    seen: dict[str, None] = {}
    for element in elements:
        for key in attribute_names(element):
            seen.setdefault(key, None)
    return list(seen.keys())
