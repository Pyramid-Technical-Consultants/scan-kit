"""Zero-field spot layout: repeated (0, 0) positions per energy layer."""

from __future__ import annotations

from typing import Any

from ..energies import STANDARD_ENERGIES_MEV
from ..generators.base import SpotLayoutGenerator, SpotRow
from ..input_map import new_layer_ids
from ..params import (
    ParamSpec,
    normalize_selected_energies,
    shared_energy_spec,
    validate_positive_int,
    validate_selected_energies,
)


class ZeroFieldLayout(SpotLayoutGenerator):
    """All spots at command position (0, 0) for each selected energy layer."""

    @property
    def id(self) -> str:
        return "zero_field_layout"

    def param_specs(self) -> list[ParamSpec]:
        return [
            shared_energy_spec(default=list(STANDARD_ENERGIES_MEV)),
            ParamSpec(
                key="spots_per_layer",
                label="Spots per Layer",
                kind="int",
                default=100,
                minimum=1,
                maximum=100_000,
                step=1,
            ),
        ]

    def validate(self, params: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        errors.extend(validate_selected_energies(params.get("selected_energies")))
        errors.extend(
            validate_positive_int(params.get("spots_per_layer"), label="Spots per Layer")
        )
        return errors

    def generate_rows(self, params: dict[str, Any]) -> list[SpotRow]:
        energies = normalize_selected_energies(
            params["selected_energies"],
            catalog=STANDARD_ENERGIES_MEV,
        )
        spots_per_layer = int(params["spots_per_layer"])
        layer_ids = new_layer_ids(len(energies))

        rows: list[SpotRow] = []
        for energy, layer_id in zip(energies, layer_ids):
            for _ in range(spots_per_layer):
                rows.append(
                    SpotRow(
                        energy=float(energy),
                        layer_id=int(layer_id),
                        x_position=0.0,
                        y_position=0.0,
                    )
                )
        return rows
