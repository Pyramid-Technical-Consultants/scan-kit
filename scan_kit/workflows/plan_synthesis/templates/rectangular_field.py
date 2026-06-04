"""Rectangular-field plan template."""

from __future__ import annotations

from ..base import PlanTemplate
from ..columns import standard_column_generators
from ..generators.base import ColumnGenerator, SpotLayoutGenerator
from ..layouts.rectangular_field import RectangularFieldLayout, rectangular_grid_positions

__all__ = ["RectangularFieldTemplate", "rectangular_grid_positions"]


class RectangularFieldTemplate(PlanTemplate):
    """Rectangular spot grid for each selected energy layer."""

    def __init__(self) -> None:
        self._layout = RectangularFieldLayout()
        self._columns = standard_column_generators()

    @property
    def id(self) -> str:
        return "rectangular_field"

    @property
    def name(self) -> str:
        return "Rectangular Field"

    @property
    def description(self) -> str:
        return "Even spot grid per layer with configurable field size."

    @property
    def layout(self) -> SpotLayoutGenerator:
        return self._layout

    @property
    def column_generators(self) -> list[ColumnGenerator]:
        return self._columns
