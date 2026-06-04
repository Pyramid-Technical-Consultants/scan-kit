"""Zero-field plan template."""

from __future__ import annotations

from ..base import PlanTemplate
from ..columns import standard_column_generators
from ..generators.base import ColumnGenerator, SpotLayoutGenerator
from ..layouts.zero_field import ZeroFieldLayout


class ZeroFieldTemplate(PlanTemplate):
    """All spots at (0, 0) for each selected energy layer."""

    def __init__(self) -> None:
        self._layout = ZeroFieldLayout()
        self._columns = standard_column_generators()

    @property
    def id(self) -> str:
        return "zero_field"

    @property
    def name(self) -> str:
        return "Zero Field"

    @property
    def description(self) -> str:
        return "Every spot at (0, 0) for each energy layer."

    @property
    def layout(self) -> SpotLayoutGenerator:
        return self._layout

    @property
    def column_generators(self) -> list[ColumnGenerator]:
        return self._columns
