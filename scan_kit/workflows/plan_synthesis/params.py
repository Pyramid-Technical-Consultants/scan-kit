"""Parameter specifications for plan synthesis templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ParamKind = Literal["energy_multiselect", "float", "int", "bool"]


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


def shared_energy_spec(*, default: list[float] | None = None) -> ParamSpec:
    return ParamSpec(
        key="selected_energies",
        label="Energy Layers (MeV)",
        kind="energy_multiselect",
        default=list(default or []),
    )


def shared_charge_spec(*, default: float = 0.02) -> ParamSpec:
    return ParamSpec(
        key="charge_req_mu",
        label="Spot Charge Request",
        kind="float",
        default=default,
        minimum=0.0001,
        maximum=1000.0,
        decimals=4,
        step=0.01,
        suffix="MU",
    )


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
    """Return catalog energies in descending order, filtered to user selection."""
    if not isinstance(energies, list):
        return []
    selected = {float(e) for e in energies}
    return [e for e in sorted(catalog, reverse=True) if e in selected]
