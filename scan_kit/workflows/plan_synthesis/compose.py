"""Compose input_map DataFrames from layout + column generators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .generators.base import ColumnGenerator, SpotLayoutGenerator
from .input_map import INPUT_MAP_COLUMNS, assemble_input_map

ProgressCallback = Callable[[int], None]


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
    *,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Run layout + column generators and return a complete input_map DataFrame."""
    def report(value: int) -> None:
        if progress is not None:
            progress(max(0, min(100, value)))

    report(0)
    rows = layout.generate_rows(params)
    report(10)

    by_column = {gen.column: gen for gen in column_generators}
    data: dict[str, list[Any]] = {}
    n_columns = len(INPUT_MAP_COLUMNS)
    for index, column in enumerate(INPUT_MAP_COLUMNS):
        generator = by_column.get(column)
        if generator is None:
            raise KeyError(f"No column generator registered for {column!r}")
        data[column] = generator.values(rows, params)
        report(10 + int(85 * (index + 1) / n_columns))

    frame = assemble_input_map(data)
    report(100)
    return frame
