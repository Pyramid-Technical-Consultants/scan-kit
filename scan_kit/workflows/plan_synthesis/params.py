"""Parameter specifications for plan synthesis templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .energies import STANDARD_ENERGIES_MEV

ParamKind = Literal["energy_multiselect", "float", "int", "bool", "choice"]
ParamFieldSet = Literal["energy", "geometry", "weight"]

PARAM_FIELD_SET_ORDER: tuple[ParamFieldSet, ...] = ("energy", "geometry", "weight")
PARAM_FIELD_SET_LABELS: dict[ParamFieldSet, str] = {
    "energy": "Energy",
    "geometry": "Geometry",
    "weight": "Weight",
}


@dataclass(frozen=True)
class ParamSpec:
    """Describes one configurable template parameter for the GUI and validator."""

    key: str
    label: str
    kind: ParamKind
    default: Any
    minimum: float | int | None = None
    maximum: float | int | None = None
    decimals: int = 3
    step: float | int | None = None
    suffix: str = ""
    row_partner: str | None = None
    sub_label: str = ""
    choices: tuple[tuple[str, str], ...] = ()
    visible_when: dict[str, tuple[Any, ...]] | None = None
    field_set: ParamFieldSet = "geometry"


def shared_energy_spec(*, default: list[float] | None = None) -> ParamSpec:
    return ParamSpec(
        key="selected_energies",
        label="Energy Layers (MeV)",
        kind="energy_multiselect",
        default=list(default if default is not None else STANDARD_ENERGIES_MEV),
        field_set="energy",
    )


SPOT_WEIGHT_LABEL = "Spot Weight (MU)"


def validate_weight_range(min_value: Any, max_value: Any) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_positive_float(min_value, label="Minimum Weight (MU)"))
    errors.extend(validate_positive_float(max_value, label="Maximum Weight (MU)"))
    if errors:
        return errors
    if float(max_value) < float(min_value):
        return [
            "Maximum Weight (MU) must be greater than or equal to Minimum Weight (MU)."
        ]
    return []

def validate_selected_energies(energies: Any) -> list[str]:
    if not isinstance(energies, list) or not energies:
        return ["Select at least one energy layer."]
    return []


def validate_positive_float(value: Any, *, label: str) -> list[str]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return [f"{label} must be a number."]
    if v <= 0:
        return [f"{label} must be greater than zero."]
    return []


def validate_positive_int(value: Any, *, label: str) -> list[str]:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return [f"{label} must be an integer."]
    if v < 1:
        return [f"{label} must be at least 1."]
    return []


def normalize_selected_energies(
    energies: Any,
    *,
    catalog: tuple[float, ...],
) -> list[float]:
    """Return selected energies in descending MeV order (highest layer first)."""
    if not isinstance(energies, list):
        return []
    selected = {float(e) for e in energies}
    return [e for e in sorted(catalog, reverse=True) if e in selected]
