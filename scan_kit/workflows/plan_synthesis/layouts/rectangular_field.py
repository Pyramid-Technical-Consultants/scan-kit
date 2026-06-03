"""Rectangular spot grid layout per energy layer."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..energies import STANDARD_ENERGIES_MEV
from ..generators.base import SpotLayoutGenerator, SpotRow
from ..input_map import new_layer_ids
from ..params import (
    ParamSpec,
    normalize_selected_energies,
    shared_energy_spec,
    validate_positive_float,
    validate_positive_int,
    validate_selected_energies,
)


def rectangular_grid_positions(
    *,
    center_x_mm: float,
    center_y_mm: float,
    field_width_mm: float,
    field_height_mm: float,
    spots_x: int,
    spots_y: int,
) -> list[tuple[float, float]]:
    """Evenly spaced grid from center ± extent/2, inclusive endpoints."""
    half_w = field_width_mm / 2.0
    half_h = field_height_mm / 2.0
    xs = np.linspace(center_x_mm - half_w, center_x_mm + half_w, spots_x)
    ys = np.linspace(center_y_mm - half_h, center_y_mm + half_h, spots_y)
    xx, yy = np.meshgrid(xs, ys)
    return list(zip(xx.ravel().tolist(), yy.ravel().tolist()))


class RectangularFieldLayout(SpotLayoutGenerator):
    """Evenly spaced rectangular grid for each selected energy layer."""

    @property
    def id(self) -> str:
        return "rectangular_field_layout"

    def param_specs(self) -> list[ParamSpec]:
        return [
            shared_energy_spec(default=list(STANDARD_ENERGIES_MEV)),
            ParamSpec(
                key="center_x_mm",
                label="Field Center",
                sub_label="X",
                kind="float",
                default=0.0,
                minimum=-500.0,
                maximum=500.0,
                decimals=3,
                step=0.1,
                suffix="mm",
            ),
            ParamSpec(
                key="center_y_mm",
                label="Field Center Y",
                row_partner="center_x_mm",
                sub_label="Y",
                kind="float",
                default=0.0,
                minimum=-500.0,
                maximum=500.0,
                decimals=3,
                step=0.1,
                suffix="mm",
            ),
            ParamSpec(
                key="field_width_mm",
                label="Field Width",
                kind="float",
                default=20.0,
                minimum=0.001,
                maximum=1000.0,
                decimals=3,
                step=0.1,
                suffix="mm",
            ),
            ParamSpec(
                key="field_height_mm",
                label="Field Height",
                kind="float",
                default=20.0,
                minimum=0.001,
                maximum=1000.0,
                decimals=3,
                step=0.1,
                suffix="mm",
            ),
            ParamSpec(
                key="spots_x",
                label="Spot Grid",
                sub_label="X",
                kind="int",
                default=10,
                minimum=1,
                maximum=1000,
                step=1,
            ),
            ParamSpec(
                key="spots_y",
                label="Spot Grid Y",
                row_partner="spots_x",
                sub_label="Y",
                kind="int",
                default=10,
                minimum=1,
                maximum=1000,
                step=1,
            ),
        ]

    def validate(self, params: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        errors.extend(validate_selected_energies(params.get("selected_energies")))
        errors.extend(
            validate_positive_float(params.get("field_width_mm"), label="Field Width")
        )
        errors.extend(
            validate_positive_float(params.get("field_height_mm"), label="Field Height")
        )
        errors.extend(validate_positive_int(params.get("spots_x"), label="Spot Grid X"))
        errors.extend(validate_positive_int(params.get("spots_y"), label="Spot Grid Y"))
        return errors

    def generate_rows(self, params: dict[str, Any]) -> list[SpotRow]:
        energies = normalize_selected_energies(
            params["selected_energies"],
            catalog=STANDARD_ENERGIES_MEV,
        )
        positions = rectangular_grid_positions(
            center_x_mm=float(params["center_x_mm"]),
            center_y_mm=float(params["center_y_mm"]),
            field_width_mm=float(params["field_width_mm"]),
            field_height_mm=float(params["field_height_mm"]),
            spots_x=int(params["spots_x"]),
            spots_y=int(params["spots_y"]),
        )
        layer_ids = new_layer_ids(len(energies))

        rows: list[SpotRow] = []
        for energy, layer_id in zip(energies, layer_ids):
            for x_pos, y_pos in positions:
                rows.append(
                    SpotRow(
                        energy=float(energy),
                        layer_id=int(layer_id),
                        x_position=float(x_pos),
                        y_position=float(y_pos),
                    )
                )
        return rows
