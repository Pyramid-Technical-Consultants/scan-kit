"""Shared spot weight (CHARGE_REQ) generation for plan templates."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Literal

import numpy as np

from .generators.base import SpotRow
from .params import (
    SPOT_WEIGHT_LABEL,
    ParamSpec,
    validate_positive_float,
    validate_weight_range,
)

SpotWeightMethod = Literal[
    "fixed",
    "random_range",
    "layer_even_range",
    "even_total",
    "random_total_variance",
]

SPOT_WEIGHT_METHOD_FIXED: SpotWeightMethod = "fixed"
SPOT_WEIGHT_METHOD_RANDOM: SpotWeightMethod = "random_range"
SPOT_WEIGHT_METHOD_LAYER_EVEN: SpotWeightMethod = "layer_even_range"
SPOT_WEIGHT_METHOD_EVEN_TOTAL: SpotWeightMethod = "even_total"
SPOT_WEIGHT_METHOD_RANDOM_TOTAL: SpotWeightMethod = "random_total_variance"

SPOT_WEIGHT_TOTAL_LABEL = "Target Total Weight (MU)"
SPOT_WEIGHT_VARIANCE_LABEL = "Spot Variance (%)"
SPOT_WEIGHT_RANGE_LABEL = "Weight Range (MU)"

SPOT_WEIGHT_METHOD_CHOICES: tuple[tuple[str, str], ...] = (
    (SPOT_WEIGHT_METHOD_FIXED, "Fixed"),
    (SPOT_WEIGHT_METHOD_RANDOM, "Random Range"),
    (SPOT_WEIGHT_METHOD_LAYER_EVEN, "Even per Layer"),
    (SPOT_WEIGHT_METHOD_EVEN_TOTAL, "Even Total"),
    (SPOT_WEIGHT_METHOD_RANDOM_TOTAL, "Random Total"),
)

_TOTAL_METHODS = (
    SPOT_WEIGHT_METHOD_EVEN_TOTAL,
    SPOT_WEIGHT_METHOD_RANDOM_TOTAL,
)

_WEIGHT_DECIMALS = 4


def spot_weight_param_specs(*, fixed_default: float = 0.02) -> list[ParamSpec]:
    """Parameter specs shared by all built-in plan templates."""
    range_methods = (SPOT_WEIGHT_METHOD_RANDOM, SPOT_WEIGHT_METHOD_LAYER_EVEN)
    return [
        ParamSpec(
            key="spot_weight_method",
            label="Spot Weight Method",
            kind="choice",
            default=SPOT_WEIGHT_METHOD_FIXED,
            choices=SPOT_WEIGHT_METHOD_CHOICES,
            field_set="weight",
        ),
        ParamSpec(
            key="spot_weight_mu",
            label=SPOT_WEIGHT_LABEL,
            kind="float",
            default=fixed_default,
            minimum=0.0001,
            maximum=1000.0,
            decimals=_WEIGHT_DECIMALS,
            step=0.01,
            visible_when={"spot_weight_method": (SPOT_WEIGHT_METHOD_FIXED,)},
            field_set="weight",
        ),
        ParamSpec(
            key="spot_weight_total_mu",
            label=SPOT_WEIGHT_TOTAL_LABEL,
            kind="float",
            default=1.0,
            minimum=0.0001,
            maximum=1_000_000.0,
            decimals=_WEIGHT_DECIMALS,
            step=0.1,
            visible_when={"spot_weight_method": _TOTAL_METHODS},
            field_set="weight",
        ),
        ParamSpec(
            key="spot_weight_variance_pct",
            label=SPOT_WEIGHT_VARIANCE_LABEL,
            kind="float",
            default=10.0,
            minimum=0.0,
            maximum=100.0,
            decimals=1,
            step=1.0,
            visible_when={"spot_weight_method": (SPOT_WEIGHT_METHOD_RANDOM_TOTAL,)},
            field_set="weight",
        ),
        ParamSpec(
            key="spot_weight_min_mu",
            label=SPOT_WEIGHT_RANGE_LABEL,
            sub_label="Min",
            kind="float",
            default=0.002,
            minimum=0.0001,
            maximum=1000.0,
            decimals=_WEIGHT_DECIMALS,
            step=0.001,
            visible_when={"spot_weight_method": range_methods},
            field_set="weight",
        ),
        ParamSpec(
            key="spot_weight_max_mu",
            label=SPOT_WEIGHT_RANGE_LABEL,
            row_partner="spot_weight_min_mu",
            sub_label="Max",
            kind="float",
            default=0.1,
            minimum=0.0001,
            maximum=1000.0,
            decimals=_WEIGHT_DECIMALS,
            step=0.001,
            visible_when={"spot_weight_method": range_methods},
            field_set="weight",
        ),
        ParamSpec(
            key="spot_weight_layer_shuffle",
            label="Shuffle Order per Layer",
            kind="bool",
            default=False,
            visible_when={"spot_weight_method": (SPOT_WEIGHT_METHOD_LAYER_EVEN,)},
            field_set="weight",
        ),
    ]


def validate_spot_weight_params(params: dict[str, Any]) -> list[str]:
    """Validate spot-weight parameters for the selected method."""
    method = params.get("spot_weight_method", SPOT_WEIGHT_METHOD_FIXED)
    if method not in {choice[0] for choice in SPOT_WEIGHT_METHOD_CHOICES}:
        return ["Select a spot weight method."]

    if method == SPOT_WEIGHT_METHOD_FIXED:
        return validate_positive_float(params.get("spot_weight_mu"), label=SPOT_WEIGHT_LABEL)

    if method in _TOTAL_METHODS:
        errors = validate_positive_float(
            params.get("spot_weight_total_mu"),
            label=SPOT_WEIGHT_TOTAL_LABEL,
        )
        if errors:
            return errors
        if method == SPOT_WEIGHT_METHOD_RANDOM_TOTAL:
            return _validate_variance_pct(params.get("spot_weight_variance_pct"))
        return []

    return validate_weight_range(
        params.get("spot_weight_min_mu"),
        params.get("spot_weight_max_mu"),
    )


def compute_spot_weights(rows: list[SpotRow], params: dict[str, Any]) -> list[float]:
    """Return one CHARGE_REQ value per spot row."""
    method = params.get("spot_weight_method", SPOT_WEIGHT_METHOD_FIXED)
    if method == SPOT_WEIGHT_METHOD_FIXED:
        weight = round(float(params["spot_weight_mu"]), _WEIGHT_DECIMALS)
        return [weight] * len(rows)

    min_mu = float(params["spot_weight_min_mu"])
    max_mu = float(params["spot_weight_max_mu"])

    if method == SPOT_WEIGHT_METHOD_RANDOM:
        return [
            round(random.uniform(min_mu, max_mu), _WEIGHT_DECIMALS) for _ in rows
        ]

    if method == SPOT_WEIGHT_METHOD_LAYER_EVEN:
        return _layer_even_weights(
            rows,
            min_mu=min_mu,
            max_mu=max_mu,
            shuffle=bool(params.get("spot_weight_layer_shuffle", False)),
        )

    if method == SPOT_WEIGHT_METHOD_EVEN_TOTAL:
        return _even_total_weights(
            len(rows),
            target_total=float(params["spot_weight_total_mu"]),
        )

    if method == SPOT_WEIGHT_METHOD_RANDOM_TOTAL:
        return _random_total_variance_weights(
            len(rows),
            target_total=float(params["spot_weight_total_mu"]),
            variance_pct=float(params["spot_weight_variance_pct"]),
        )

    raise ValueError(f"Unsupported spot weight method: {method!r}")


def _layer_even_weights(
    rows: list[SpotRow],
    *,
    min_mu: float,
    max_mu: float,
    shuffle: bool = False,
) -> list[float]:
    layer_indices: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        layer_indices[row.layer_id].append(index)

    weights = [0.0] * len(rows)
    for indices in layer_indices.values():
        values = [
            round(float(value), _WEIGHT_DECIMALS)
            for value in np.linspace(min_mu, max_mu, len(indices))
        ]
        if shuffle and len(values) > 1:
            random.shuffle(values)
        for index, value in zip(indices, values):
            weights[index] = value
    return weights


def _validate_variance_pct(value: Any) -> list[str]:
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return [f"{SPOT_WEIGHT_VARIANCE_LABEL} must be a number."]
    if pct < 0:
        return [f"{SPOT_WEIGHT_VARIANCE_LABEL} must be zero or greater."]
    if pct > 100:
        return [f"{SPOT_WEIGHT_VARIANCE_LABEL} must be at most 100."]
    return []


def _apply_total_remainder(weights: list[float], *, target_total: float) -> list[float]:
    remainder = round(target_total - sum(weights), _WEIGHT_DECIMALS)
    if remainder:
        weights[-1] = round(weights[-1] + remainder, _WEIGHT_DECIMALS)
    return weights


def _random_total_variance_weights(
    n_spots: int,
    *,
    target_total: float,
    variance_pct: float,
) -> list[float]:
    """Random per-spot weights around the mean, scaled to *target_total*."""
    if n_spots <= 0:
        return []
    if variance_pct <= 0:
        return _even_total_weights(n_spots, target_total=target_total)

    base = target_total / n_spots
    spread = variance_pct / 100.0
    multipliers = [1.0 + random.uniform(-spread, spread) for _ in range(n_spots)]
    raw = [base * multiplier for multiplier in multipliers]
    scaled_total = sum(raw)
    if scaled_total <= 0:
        raise ValueError(
            f"{SPOT_WEIGHT_VARIANCE_LABEL} is too large to assign positive spot weights."
        )

    scale = target_total / scaled_total
    weights = [round(value * scale, _WEIGHT_DECIMALS) for value in raw]
    weights = _apply_total_remainder(weights, target_total=target_total)
    if weights[-1] <= 0:
        raise ValueError(
            f"{SPOT_WEIGHT_TOTAL_LABEL} is too small to assign a positive weight "
            f"to each of {n_spots} spots at {variance_pct:g}% variance."
        )
    return weights


def _even_total_weights(n_spots: int, *, target_total: float) -> list[float]:
    """Split *target_total* evenly across spots; last spot absorbs rounding remainder."""
    if n_spots <= 0:
        return []
    per_spot = target_total / n_spots
    weights = [round(per_spot, _WEIGHT_DECIMALS)] * n_spots
    weights = _apply_total_remainder(weights, target_total=target_total)
    if weights[-1] <= 0:
        raise ValueError(
            f"{SPOT_WEIGHT_TOTAL_LABEL} is too small to assign a positive weight "
            f"to each of {n_spots} spots."
        )
    return weights
