"""CHARGE_REQ column generator."""

from __future__ import annotations

from typing import Any

from ..params import ParamSpec
from ..spot_weight import compute_spot_weights, spot_weight_param_specs, validate_spot_weight_params
from .base import ColumnGenerator, SpotRow


class SpotWeightGenerator(ColumnGenerator):
    """Spot weight (MU) column using the shared spot-weight engine."""

    @property
    def column(self) -> str:
        return "CHARGE_REQ"

    def param_specs(self) -> list[ParamSpec]:
        return spot_weight_param_specs()

    def validate(self, params: dict[str, Any]) -> list[str]:
        return validate_spot_weight_params(params)

    def values(self, rows: list[SpotRow], params: dict[str, Any]) -> list[Any]:
        return compute_spot_weights(rows, params)


# Backward-compatible alias
UniformChargeGenerator = SpotWeightGenerator
