"""Rectangular-field plan template."""

from __future__ import annotations

from ..base import PlanTemplate
from ..columns import standard_column_generators
from ..generators.base import ColumnGenerator, SpotLayoutGenerator
from ..layouts.rectangular_field import (
    FAST_AXIS_X,
    FAST_AXIS_Y,
    LAYER_TRANSITION_CONTINUE,
    LAYER_TRANSITION_RESET,
    START_CORNER_BOTTOM_LEFT,
    START_CORNER_BOTTOM_RIGHT,
    START_CORNER_TOP_LEFT,
    START_CORNER_TOP_RIGHT,
    RectangularFieldLayout,
    positions_for_layer,
    rectangular_grid_positions,
)

__all__ = [
    "FAST_AXIS_X",
    "FAST_AXIS_Y",
    "LAYER_TRANSITION_CONTINUE",
    "LAYER_TRANSITION_RESET",
    "START_CORNER_BOTTOM_LEFT",
    "START_CORNER_BOTTOM_RIGHT",
    "START_CORNER_TOP_LEFT",
    "START_CORNER_TOP_RIGHT",
    "RectangularFieldTemplate",
    "positions_for_layer",
    "rectangular_grid_positions",
]


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
