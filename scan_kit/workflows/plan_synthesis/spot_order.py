"""Spot delivery ordering helpers for plan synthesis."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .layouts.rectangular_field import FAST_AXIS_X, FAST_AXIS_Y

SPOT_ORDER_PLAN = "plan_order"
SPOT_ORDER_MINIMIZE_TRAVEL = "minimize_travel"

SPOT_ORDER_CHOICES: tuple[tuple[str, str], ...] = (
    (SPOT_ORDER_PLAN, "Plan Order"),
    (SPOT_ORDER_MINIMIZE_TRAVEL, "Minimize Travel"),
)

_VALID_SPOT_ORDERS = {SPOT_ORDER_PLAN, SPOT_ORDER_MINIMIZE_TRAVEL}
_VALID_FAST_AXES = {FAST_AXIS_X, FAST_AXIS_Y}


def validate_spot_order_params(
    spot_order: object,
    *,
    fast_axis: object,
) -> list[str]:
    errors: list[str] = []
    order = str(spot_order or SPOT_ORDER_PLAN)
    if order not in _VALID_SPOT_ORDERS:
        errors.append("Spot order must be Plan Order or Minimize Travel.")
    if order == SPOT_ORDER_MINIMIZE_TRAVEL:
        axis = str(fast_axis or FAST_AXIS_X)
        if axis not in _VALID_FAST_AXES:
            errors.append("Fast axis must be X or Y.")
    return errors


def order_spot_indices(
    x: Sequence[float],
    y: Sequence[float],
    *,
    plan_indices: Sequence[int],
    spot_order: str = SPOT_ORDER_PLAN,
    fast_axis: str = FAST_AXIS_X,
) -> list[int]:
    """Return spot indices in the requested delivery order."""
    n = len(x)
    if n <= 1:
        return list(range(n))

    if spot_order == SPOT_ORDER_PLAN:
        return sorted(range(n), key=lambda index: plan_indices[index])

    return serpentine_order_indices(x, y, fast_axis=fast_axis)


def serpentine_order_indices(
    x: Sequence[float],
    y: Sequence[float],
    *,
    fast_axis: str = FAST_AXIS_X,
) -> list[int]:
    """Order spots in a layer with a serpentine raster along the fast axis."""
    n = len(x)
    if n <= 1:
        return list(range(n))

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if fast_axis == FAST_AXIS_X:
        slow = y_arr
        fast = x_arr
    elif fast_axis == FAST_AXIS_Y:
        slow = x_arr
        fast = y_arr
    else:
        raise ValueError(f"Unsupported fast axis: {fast_axis!r}")

    slow_order = np.argsort(slow, kind="stable")
    tolerance = _row_tolerance(slow[slow_order])

    rows: list[list[int]] = []
    current_row = [int(slow_order[0])]
    for spot_index in slow_order[1:]:
        spot_index = int(spot_index)
        if abs(float(slow[spot_index]) - float(slow[current_row[-1]])) <= tolerance:
            current_row.append(spot_index)
        else:
            rows.append(current_row)
            current_row = [spot_index]
    rows.append(current_row)

    ordered: list[int] = []
    for row_index, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda index: float(fast[index]))
        if row_index % 2 == 1:
            row_sorted.reverse()
        ordered.extend(row_sorted)
    return ordered


def _row_tolerance(sorted_slow_coords: np.ndarray) -> float:
    unique = np.unique(sorted_slow_coords)
    if len(unique) <= 1:
        return 1.0
    diffs = np.diff(unique)
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return 1.0
    return float(np.min(positive)) * 0.5
