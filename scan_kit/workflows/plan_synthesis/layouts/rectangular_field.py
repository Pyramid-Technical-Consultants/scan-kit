"""Rectangular spot grid layout per energy layer."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..energies import STANDARD_ENERGIES_MEV
from ..generators.base import SpotLayoutGenerator, SpotRow
from ..input_map import new_layer_ids
from ..params import (
    ParamSpec,
    QuickSet,
    normalize_selected_energies,
    shared_energy_spec,
    validate_positive_float,
    validate_positive_int,
    validate_selected_energies,
)

_FIELD_SIZE_QUICK_SETS: tuple[QuickSet, ...] = tuple(
    QuickSet(
        label=str(size_mm),
        values={"field_width_mm": float(size_mm), "field_height_mm": float(size_mm)},
    )
    for size_mm in (100, 200, 250, 300)
)

_SPOT_GRID_QUICK_SETS: tuple[QuickSet, ...] = tuple(
    QuickSet(
        label=str(count),
        values={"spots_x": count, "spots_y": count},
    )
    for count in (3, 7, 11, 33)
)

FAST_AXIS_X = "x"
FAST_AXIS_Y = "y"
FAST_AXIS_CHOICES: tuple[tuple[str, str], ...] = (
    (FAST_AXIS_X, "X"),
    (FAST_AXIS_Y, "Y"),
)

START_CORNER_TOP_LEFT = "top_left"
START_CORNER_TOP_RIGHT = "top_right"
START_CORNER_BOTTOM_LEFT = "bottom_left"
START_CORNER_BOTTOM_RIGHT = "bottom_right"
START_CORNER_CHOICES: tuple[tuple[str, str], ...] = (
    (START_CORNER_TOP_LEFT, "Top Left"),
    (START_CORNER_TOP_RIGHT, "Top Right"),
    (START_CORNER_BOTTOM_LEFT, "Bottom Left"),
    (START_CORNER_BOTTOM_RIGHT, "Bottom Right"),
)
_VALID_START_CORNERS = {
    START_CORNER_TOP_LEFT,
    START_CORNER_TOP_RIGHT,
    START_CORNER_BOTTOM_LEFT,
    START_CORNER_BOTTOM_RIGHT,
}

LAYER_TRANSITION_RESET = "reset"
LAYER_TRANSITION_CONTINUE = "continue"
LAYER_TRANSITION_CHOICES: tuple[tuple[str, str], ...] = (
    (LAYER_TRANSITION_RESET, "Reset to Corner"),
    (LAYER_TRANSITION_CONTINUE, "Continue from End"),
)
_VALID_LAYER_TRANSITIONS = {
    LAYER_TRANSITION_RESET,
    LAYER_TRANSITION_CONTINUE,
}


def positions_for_layer(
    base_positions: list[tuple[float, float]],
    layer_index: int,
    *,
    layer_transition: str,
) -> list[tuple[float, float]]:
    """Return raster order for one energy layer."""
    if layer_transition == LAYER_TRANSITION_RESET:
        return base_positions
    if layer_index % 2 == 0:
        return base_positions
    return list(reversed(base_positions))


def rectangular_grid_positions(
    *,
    center_x_mm: float,
    center_y_mm: float,
    field_width_mm: float,
    field_height_mm: float,
    spots_x: int,
    spots_y: int,
    fast_axis: str = FAST_AXIS_X,
    start_corner: str = START_CORNER_TOP_LEFT,
) -> list[tuple[float, float]]:
    """Evenly spaced grid with serpentine raster along the fast axis."""
    half_w = field_width_mm / 2.0
    half_h = field_height_mm / 2.0
    xs = np.linspace(center_x_mm - half_w, center_x_mm + half_w, spots_x)
    ys = np.linspace(center_y_mm - half_h, center_y_mm + half_h, spots_y)

    positions: list[tuple[float, float]] = []
    if fast_axis == FAST_AXIS_Y:
        start_at_low_x = start_corner in {
            START_CORNER_BOTTOM_LEFT,
            START_CORNER_TOP_LEFT,
        }
        start_y_forward = start_corner in {
            START_CORNER_BOTTOM_LEFT,
            START_CORNER_BOTTOM_RIGHT,
        }
        slow_vals = list(xs) if start_at_low_x else list(xs)[::-1]
        for slow_idx, x in enumerate(slow_vals):
            y_forward = start_y_forward if slow_idx % 2 == 0 else not start_y_forward
            y_vals = ys if y_forward else ys[::-1]
            for y in y_vals:
                positions.append((float(x), float(y)))
        return positions

    start_at_low_y = start_corner in {
        START_CORNER_BOTTOM_LEFT,
        START_CORNER_BOTTOM_RIGHT,
    }
    start_x_forward = start_corner in {
        START_CORNER_BOTTOM_LEFT,
        START_CORNER_TOP_LEFT,
    }
    slow_vals = list(ys) if start_at_low_y else list(ys)[::-1]
    for slow_idx, y in enumerate(slow_vals):
        x_forward = start_x_forward if slow_idx % 2 == 0 else not start_x_forward
        x_vals = xs if x_forward else xs[::-1]
        for x in x_vals:
            positions.append((float(x), float(y)))
    return positions


class RectangularFieldLayout(SpotLayoutGenerator):
    """Evenly spaced rectangular grid for each selected energy layer."""

    @property
    def id(self) -> str:
        return "rectangular_field_layout"

    def param_specs(self) -> list[ParamSpec]:
        return [
            shared_energy_spec(),
            ParamSpec(
                key="center_x_mm",
                label="Field Center (mm)",
                sub_label="X",
                kind="float",
                default=0.0,
                minimum=-500.0,
                maximum=500.0,
                decimals=3,
                step=0.1,
                field_set="geometry",
                quick_sets=(
                    QuickSet(
                        label="0,0",
                        values={"center_x_mm": 0.0, "center_y_mm": 0.0},
                    ),
                ),
            ),
            ParamSpec(
                key="center_y_mm",
                label="Field Center Y (mm)",
                row_partner="center_x_mm",
                sub_label="Y",
                kind="float",
                default=0.0,
                minimum=-500.0,
                maximum=500.0,
                decimals=3,
                step=0.1,
                field_set="geometry",
            ),
            ParamSpec(
                key="field_width_mm",
                label="Field Size (mm)",
                sub_label="W",
                kind="float",
                default=100.0,
                minimum=0.001,
                maximum=1000.0,
                decimals=3,
                step=0.1,
                field_set="geometry",
                quick_sets=_FIELD_SIZE_QUICK_SETS,
            ),
            ParamSpec(
                key="field_height_mm",
                label="Field Size H (mm)",
                row_partner="field_width_mm",
                sub_label="H",
                kind="float",
                default=100.0,
                minimum=0.001,
                maximum=1000.0,
                decimals=3,
                step=0.1,
                field_set="geometry",
            ),
            ParamSpec(
                key="spots_x",
                label="Spot Grid (spots)",
                sub_label="X",
                kind="int",
                default=33,
                minimum=1,
                maximum=1000,
                step=1,
                field_set="geometry",
                quick_sets=_SPOT_GRID_QUICK_SETS,
            ),
            ParamSpec(
                key="spots_y",
                label="Spot Grid Y",
                row_partner="spots_x",
                sub_label="Y",
                kind="int",
                default=33,
                minimum=1,
                maximum=1000,
                step=1,
                field_set="geometry",
            ),
            ParamSpec(
                key="fast_axis",
                label="Fast Axis",
                kind="button_group",
                default=FAST_AXIS_X,
                choices=FAST_AXIS_CHOICES,
                field_set="geometry",
            ),
            ParamSpec(
                key="start_corner",
                label="Start Corner",
                kind="button_group",
                default=START_CORNER_TOP_LEFT,
                choices=START_CORNER_CHOICES,
                field_set="geometry",
            ),
            ParamSpec(
                key="layer_transition",
                label="Layer Transition",
                kind="button_group",
                default=LAYER_TRANSITION_RESET,
                choices=LAYER_TRANSITION_CHOICES,
                field_set="geometry",
            ),
        ]

    def validate(self, params: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        errors.extend(validate_selected_energies(params.get("selected_energies")))
        errors.extend(
            validate_positive_float(
                params.get("field_width_mm"), label="Field Size (mm) W"
            )
        )
        errors.extend(
            validate_positive_float(
                params.get("field_height_mm"), label="Field Size (mm) H"
            )
        )
        errors.extend(
            validate_positive_int(params.get("spots_x"), label="Spot Grid (spots) X")
        )
        errors.extend(
            validate_positive_int(params.get("spots_y"), label="Spot Grid (spots) Y")
        )
        fast_axis = params.get("fast_axis", FAST_AXIS_X)
        if fast_axis not in {FAST_AXIS_X, FAST_AXIS_Y}:
            errors.append("Fast axis must be X or Y.")
        start_corner = params.get("start_corner", START_CORNER_TOP_LEFT)
        if start_corner not in _VALID_START_CORNERS:
            errors.append("Start corner must be Top Left, Top Right, Bottom Left, or Bottom Right.")
        layer_transition = params.get("layer_transition", LAYER_TRANSITION_RESET)
        if layer_transition not in _VALID_LAYER_TRANSITIONS:
            errors.append(
                "Layer transition must be Reset to Corner or Continue from End."
            )
        return errors

    def generate_rows(self, params: dict[str, Any]) -> list[SpotRow]:
        energies = normalize_selected_energies(
            params["selected_energies"],
            catalog=STANDARD_ENERGIES_MEV,
        )
        base_positions = rectangular_grid_positions(
            center_x_mm=float(params["center_x_mm"]),
            center_y_mm=float(params["center_y_mm"]),
            field_width_mm=float(params["field_width_mm"]),
            field_height_mm=float(params["field_height_mm"]),
            spots_x=int(params["spots_x"]),
            spots_y=int(params["spots_y"]),
            fast_axis=str(params.get("fast_axis", FAST_AXIS_X)),
            start_corner=str(params.get("start_corner", START_CORNER_TOP_LEFT)),
        )
        layer_transition = str(
            params.get("layer_transition", LAYER_TRANSITION_RESET)
        )
        layer_ids = new_layer_ids(len(energies))

        rows: list[SpotRow] = []
        for layer_index, (energy, layer_id) in enumerate(zip(energies, layer_ids)):
            positions = positions_for_layer(
                base_positions,
                layer_index,
                layer_transition=layer_transition,
            )
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
