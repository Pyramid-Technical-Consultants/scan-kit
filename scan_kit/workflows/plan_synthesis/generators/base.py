"""Column and layout generator base types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..params import ParamSpec


@dataclass(frozen=True)
class SpotRow:
    """One spot in the plan before column generators fill the input_map row."""

    energy: float
    layer_id: int
    x_position: float
    y_position: float


class ColumnGenerator(ABC):
    """Generates values for one input_map.csv column across all spot rows."""

    @property
    @abstractmethod
    def column(self) -> str:
        """Target column name (empty string for the trailing CSV column)."""

    def param_specs(self) -> list[ParamSpec]:
        return []

    def validate(self, params: dict[str, Any]) -> list[str]:
        return []

    @abstractmethod
    def values(self, rows: list[SpotRow], params: dict[str, Any]) -> list[Any]:
        """Return one value per spot row."""


class SpotLayoutGenerator(ABC):
    """Builds the spot row list (energy layers + geometry + layer IDs)."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Stable layout identifier."""

    @abstractmethod
    def param_specs(self) -> list[ParamSpec]:
        """User-facing parameters for this layout."""

    @abstractmethod
    def validate(self, params: dict[str, Any]) -> list[str]:
        """Return human-readable validation errors."""

    @abstractmethod
    def generate_rows(self, params: dict[str, Any]) -> list[SpotRow]:
        """Produce spot rows in delivery order."""
