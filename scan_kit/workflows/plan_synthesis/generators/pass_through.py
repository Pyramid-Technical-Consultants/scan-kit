"""Pass-through column generators reading fields from SpotRow."""

from __future__ import annotations

from typing import Any

from .base import ColumnGenerator, SpotRow


class FromRowColumnGenerator(ColumnGenerator):
    """Expose one :class:`SpotRow` field as an input_map column."""

    def __init__(self, column: str, field: str) -> None:
        self._column = column
        self._field = field

    @property
    def column(self) -> str:
        return self._column

    def values(self, rows: list[SpotRow], params: dict[str, Any]) -> list[Any]:
        del params
        return [getattr(row, self._field) for row in rows]
