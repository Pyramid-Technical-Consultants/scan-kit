"""Sequential global spot_no column generator."""

from __future__ import annotations

from typing import Any

from .base import ColumnGenerator, SpotRow


class SequentialSpotNoGenerator(ColumnGenerator):
    """Assign spot_no 0..N-1 in row order."""

    @property
    def column(self) -> str:
        return "spot_no"

    def values(self, rows: list[SpotRow], params: dict[str, Any]) -> list[Any]:
        del params
        return list(range(len(rows)))
