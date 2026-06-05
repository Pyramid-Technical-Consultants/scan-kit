"""Shared helpers for external plan import templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .input_map import (
    DEFAULT_CURRENT_A,
    assemble_input_map,
    new_layer_ids,
)
from .layouts.rectangular_field import FAST_AXIS_X
from .spot_order import SPOT_ORDER_PLAN, order_spot_indices


@dataclass(frozen=True)
class ImportSpot:
    """One spot extracted from an external plan before input_map assembly."""

    x: float
    y: float
    energy: float
    charge: float
    beam_size: float
    plan_index: int


class PlanImportError(Exception):
    """Raised when an external plan file cannot be parsed."""


class ImportSpotLike(Protocol):
    x: float
    y: float
    energy: float
    charge: float
    beam_size: float
    plan_index: int


def order_spots_within_layers(
    spots: list[ImportSpotLike],
    *,
    spot_order: str = SPOT_ORDER_PLAN,
    fast_axis: str = FAST_AXIS_X,
) -> list[ImportSpotLike]:
    """Reorder spots within each energy layer."""
    by_energy: dict[float, list[ImportSpotLike]] = {}
    energy_order: list[float] = []
    for spot in spots:
        if spot.energy not in by_energy:
            by_energy[spot.energy] = []
            energy_order.append(spot.energy)
        by_energy[spot.energy].append(spot)

    ordered: list[ImportSpotLike] = []
    for energy in energy_order:
        layer_spots = by_energy[energy]
        indices = order_spot_indices(
            [spot.x for spot in layer_spots],
            [spot.y for spot in layer_spots],
            plan_indices=[spot.plan_index for spot in layer_spots],
            spot_order=spot_order,
            fast_axis=fast_axis,
        )
        ordered.extend(layer_spots[index] for index in indices)
    return ordered


def import_spots_to_input_map(
    spots: list[ImportSpotLike],
    *,
    default_current: float = DEFAULT_CURRENT_A,
    spot_order: str = SPOT_ORDER_PLAN,
    fast_axis: str = FAST_AXIS_X,
) -> pd.DataFrame:
    """Convert imported spots into an input_map DataFrame."""
    if not spots:
        raise PlanImportError("No planned spots found in plan file")

    ordered_spots = order_spots_within_layers(
        spots,
        spot_order=spot_order,
        fast_axis=fast_axis,
    )

    unique_energies = list(dict.fromkeys(spot.energy for spot in ordered_spots))
    layer_ids = new_layer_ids(len(unique_energies))
    energy_to_layer = dict(zip(unique_energies, layer_ids))

    data = {
        "ENERGY": [spot.energy for spot in ordered_spots],
        "CURRENT": [default_current] * len(ordered_spots),
        "BEAM_SIZE": [spot.beam_size for spot in ordered_spots],
        "X_POSITION": [spot.x for spot in ordered_spots],
        "Y_POSITION": [spot.y for spot in ordered_spots],
        "CHARGE_REQ": [spot.charge for spot in ordered_spots],
        "VELOCITY": [0.0] * len(ordered_spots),
        "spot_no": list(range(len(ordered_spots))),
        "layer_id": [energy_to_layer[spot.energy] for spot in ordered_spots],
    }
    return assemble_input_map(data)
