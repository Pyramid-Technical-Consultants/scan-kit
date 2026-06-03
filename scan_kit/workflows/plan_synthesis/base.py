"""Plan synthesis template base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from .compose import collect_param_specs, generate_input_map, validate_params
from .generators.base import ColumnGenerator, SpotLayoutGenerator
from .params import ParamSpec


class PlanTemplate(ABC):
    """Built-in plan template: spot layout + per-column generators."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Stable template identifier."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name in the template list."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description shown above the parameter form."""

    @property
    @abstractmethod
    def layout(self) -> SpotLayoutGenerator:
        """Spot geometry / energy layer layout generator."""

    @property
    @abstractmethod
    def column_generators(self) -> list[ColumnGenerator]:
        """One generator per input_map column."""

    def param_specs(self) -> list[ParamSpec]:
        return collect_param_specs(self.layout, self.column_generators)

    def default_params(self) -> dict[str, Any]:
        return {spec.key: spec.default for spec in self.param_specs()}

    def validate(self, params: dict[str, Any]) -> list[str]:
        return validate_params(self.layout, self.column_generators, params)

    def generate(self, params: dict[str, Any]) -> pd.DataFrame:
        return generate_input_map(self.layout, self.column_generators, params)
