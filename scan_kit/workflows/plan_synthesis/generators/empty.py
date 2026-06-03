"""Trailing empty CSV column generator."""

from __future__ import annotations

from typing import Any

from .base import ColumnGenerator, SpotRow


class EmptyTrailingColumnGenerator(ColumnGenerator):
    """Preserve the trailing comma column present in reference input_map.csv files."""

    @property
    def column(self) -> str:
        return ""

    def values(self, rows: list[SpotRow], params: dict[str, Any]) -> list[Any]:
        del params
        return [""] * len(rows)
