"""Compose input_map DataFrames from layout + column generators."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .generators.base import ColumnGenerator, SpotLayoutGenerator
from .input_map import INPUT_MAP_COLUMNS, assemble_input_map


def collect_param_specs(
    layout: SpotLayoutGenerator,
    column_generators: list[ColumnGenerator],
) -> list:
    """Merge param specs from layout and column generators (dedupe by key)."""
    from .params import ParamSpec

    specs: list[ParamSpec] = []
    seen: set[str] = set()
    for source in (layout, *column_generators):
        for spec in source.param_specs():
            if spec.key not in seen:
                specs.append(spec)
                seen.add(spec.key)
    return specs


def validate_params(
    layout: SpotLayoutGenerator,
    column_generators: list[ColumnGenerator],
    params: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    errors.extend(layout.validate(params))
    for gen in column_generators:
        errors.extend(gen.validate(params))
    return errors


def generate_input_map(
    layout: SpotLayoutGenerator,
    column_generators: list[ColumnGenerator],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Run layout + column generators and return a complete input_map DataFrame."""
    rows = layout.generate_rows(params)
    by_column = {gen.column: gen for gen in column_generators}
    data: dict[str, list[Any]] = {}
    for column in INPUT_MAP_COLUMNS:
        generator = by_column.get(column)
        if generator is None:
            raise KeyError(f"No column generator registered for {column!r}")
        data[column] = generator.values(rows, params)
    return assemble_input_map(data)
