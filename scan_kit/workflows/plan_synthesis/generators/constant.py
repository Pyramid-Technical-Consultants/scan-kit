"""Constant-value column generator."""

from __future__ import annotations

from typing import Any

from .base import ColumnGenerator, SpotRow


class ConstantColumnGenerator(ColumnGenerator):
    """Fill a column with a fixed value on every row."""

    def __init__(self, column: str, value: float | int) -> None:
        self._column = column
        self._value = value

    @property
    def column(self) -> str:
        return self._column

    def values(self, rows: list[SpotRow], params: dict[str, Any]) -> list[Any]:
        del params
        return [self._value] * len(rows)
