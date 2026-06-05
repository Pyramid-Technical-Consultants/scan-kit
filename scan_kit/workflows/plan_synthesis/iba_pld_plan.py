"""Load IBA PBS PLD plans and convert them to input_map rows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .input_map import DEFAULT_BEAM_SIZE, DEFAULT_CURRENT_A
from .layouts.rectangular_field import FAST_AXIS_X
from .plan_import import ImportSpot, PlanImportError, import_spots_to_input_map
from .spot_order import SPOT_ORDER_PLAN


class PldPlanError(PlanImportError):
    """Raised when an IBA PLD plan cannot be parsed."""


@dataclass(frozen=True)
class _PldBeamHeader:
    plan_label: str
    beam_name: str
    total_mu: float
    cumulative_weight: float
    num_layers: int


@dataclass(frozen=True)
class _PldLayerHeader:
    spot_id: str
    energy: float
    cumulative_weight: float
    num_elements: int
    num_paintings: int


def is_iba_pld_file(path: Path) -> bool:
    """Return True when *path* looks like an IBA PLD plan file."""
    return path.is_file() and path.suffix.lower() == ".pld"


def pld_plan_label_from_path(path: Path) -> str:
    """Return the plan label from a PLD beam header, or the file stem."""
    header = _parse_pld_text(path.read_text(encoding="utf-8-sig"))[0]
    return header.plan_label or path.stem


def _parse_float(value: str, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise PldPlanError(f"Invalid {label}: {value!r}") from None
    if not math.isfinite(number):
        raise PldPlanError(f"Invalid {label}: {value!r}")
    return number


def _parse_int(value: str, *, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise PldPlanError(f"Invalid {label}: {value!r}") from None
    return number


def _parse_beam_header(line: str) -> _PldBeamHeader:
    parts = [part.strip() for part in line.split(",")]
    if not parts or parts[0] != "Beam":
        raise PldPlanError("PLD file must start with a Beam header line")
    if len(parts) < 10:
        raise PldPlanError("PLD Beam header is missing required fields")

    return _PldBeamHeader(
        plan_label=parts[5],
        beam_name=parts[6],
        total_mu=_parse_float(parts[7], label="beam total MU"),
        cumulative_weight=_parse_float(parts[8], label="cumulative meterset weight"),
        num_layers=_parse_int(parts[9], label="layer count"),
    )


def _parse_layer_header(line: str) -> _PldLayerHeader:
    parts = [part.strip() for part in line.split(",")]
    if not parts or parts[0] != "Layer":
        raise PldPlanError("Expected a Layer header line")
    if len(parts) < 5:
        raise PldPlanError("PLD Layer header is missing required fields")

    num_paintings = 1
    if len(parts) >= 6:
        num_paintings = _parse_int(parts[5], label="number of paintings")

    return _PldLayerHeader(
        spot_id=parts[1],
        energy=_parse_float(parts[2], label="layer energy"),
        cumulative_weight=_parse_float(parts[3], label="layer cumulative weight"),
        num_elements=_parse_int(parts[4], label="element count"),
        num_paintings=num_paintings,
    )


def _meterset_weight_to_mu(
    weight: float,
    *,
    total_mu: float,
    cumulative_weight: float,
) -> float:
    if weight <= 0.0 or cumulative_weight <= 0.0:
        return 0.0
    return weight * total_mu / cumulative_weight


def _spots_from_pld_text(
    text: str,
    *,
    default_beam_size: float,
) -> tuple[_PldBeamHeader, list[ImportSpot]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise PldPlanError("PLD file is empty")

    beam_header = _parse_beam_header(lines[0])
    spots: list[ImportSpot] = []
    plan_index = 0
    current_layer: _PldLayerHeader | None = None
    pending_weights: dict[tuple[float, float], float] = {}

    def flush_layer() -> None:
        nonlocal plan_index
        if current_layer is None:
            return
        for (x_pos, y_pos), weight in pending_weights.items():
            charge = _meterset_weight_to_mu(
                weight,
                total_mu=beam_header.total_mu,
                cumulative_weight=beam_header.cumulative_weight,
            )
            if charge <= 0.0:
                continue
            spots.append(
                ImportSpot(
                    x=x_pos,
                    y=y_pos,
                    energy=current_layer.energy,
                    charge=charge,
                    beam_size=default_beam_size,
                    plan_index=plan_index,
                )
            )
            plan_index += 1
        pending_weights.clear()

    for line in lines[1:]:
        if line.startswith("Layer,"):
            flush_layer()
            current_layer = _parse_layer_header(line)
            continue
        if not line.startswith("Element,"):
            raise PldPlanError(f"Unexpected PLD line: {line!r}")
        if current_layer is None:
            raise PldPlanError("PLD Element line found before any Layer header")

        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            raise PldPlanError(f"Invalid PLD element line: {line!r}")

        x_pos = _parse_float(parts[1], label="element X position")
        y_pos = _parse_float(parts[2], label="element Y position")
        weight = _parse_float(parts[3], label="element meterset weight")
        key = (x_pos, y_pos)
        pending_weights[key] = pending_weights.get(key, 0.0) + weight

    flush_layer()

    if current_layer is None:
        raise PldPlanError("PLD file contains no Layer headers")
    return beam_header, spots


def _parse_pld_text(text: str) -> tuple[_PldBeamHeader, list[ImportSpot]]:
    return _spots_from_pld_text(text, default_beam_size=DEFAULT_BEAM_SIZE)


def pld_to_input_map(
    path: str | Path,
    *,
    default_beam_size: float = DEFAULT_BEAM_SIZE,
    default_current: float = DEFAULT_CURRENT_A,
    spot_order: str = SPOT_ORDER_PLAN,
    fast_axis: str = FAST_AXIS_X,
) -> pd.DataFrame:
    """Parse an IBA PLD plan and return an input_map DataFrame."""
    plan_path = Path(path)
    if not plan_path.is_file():
        raise PldPlanError(f"PLD plan file not found: {plan_path}")

    try:
        text = plan_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise PldPlanError(f"Could not read PLD plan file: {plan_path}") from exc

    _header, spots = _spots_from_pld_text(text, default_beam_size=default_beam_size)
    if not spots:
        raise PldPlanError("No planned spots with positive MU found in PLD plan")

    return import_spots_to_input_map(
        spots,
        default_current=default_current,
        spot_order=spot_order,
        fast_axis=fast_axis,
    )


def validate_pld_plan_path(path: Any) -> list[str]:
    text = str(path or "").strip()
    if not text:
        return ["Select an IBA PLD plan file (.pld)."]
    plan_path = Path(text)
    if not plan_path.is_file():
        return [f"PLD plan file not found: {plan_path}"]
    if plan_path.suffix.lower() != ".pld":
        return ["Plan file must have a .pld extension."]
    return []
