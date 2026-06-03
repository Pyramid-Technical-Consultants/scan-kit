"""CHARGE_REQ column generator."""

from __future__ import annotations

from typing import Any

from ..params import ParamSpec, shared_charge_spec, validate_positive_float
from .base import ColumnGenerator, SpotRow


class UniformChargeGenerator(ColumnGenerator):
    """Same MU charge request on every spot."""

    @property
    def column(self) -> str:
        return "CHARGE_REQ"

    def param_specs(self) -> list[ParamSpec]:
        return [shared_charge_spec()]

    def validate(self, params: dict[str, Any]) -> list[str]:
        return validate_positive_float(params.get("charge_req_mu"), label="Spot Charge Request")

    def values(self, rows: list[SpotRow], params: dict[str, Any]) -> list[Any]:
        charge = float(params["charge_req_mu"])
        return [charge] * len(rows)
