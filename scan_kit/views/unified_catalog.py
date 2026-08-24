"""Shared catalog types for unified Qt analysis views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DATA_SOURCE_SPOT = "spot"
DATA_SOURCE_TIMESLICE = "timeslice"

DataSourceKind = Literal["spot", "timeslice"]

DATA_SOURCES: tuple[tuple[str, str], ...] = (
    (DATA_SOURCE_SPOT, "Spot"),
    (DATA_SOURCE_TIMESLICE, "Timeslice"),
)


@dataclass(frozen=True)
class UnifiedViewOption:
    """One selectable analysis/plot option within a unified viewer."""

    id: str
    label: str
    source: DataSourceKind


def options_for_source(
    options: tuple[UnifiedViewOption, ...],
    source: DataSourceKind,
) -> tuple[UnifiedViewOption, ...]:
    return tuple(opt for opt in options if opt.source == source)


def is_option_available(
    availability: dict[str, bool],
    option: UnifiedViewOption,
) -> bool:
    composite = option_key(option.source, option.id)
    if composite in availability:
        return availability[composite]
    return availability.get(option.id, False)


def source_has_available_options(
    options: tuple[UnifiedViewOption, ...],
    availability: dict[str, bool],
    source: DataSourceKind,
) -> bool:
    return any(
        is_option_available(availability, opt)
        for opt in options_for_source(options, source)
    )


def default_source(
    options: tuple[UnifiedViewOption, ...],
    availability: dict[str, bool],
) -> DataSourceKind:
    for source, _label in DATA_SOURCES:
        if source_has_available_options(options, availability, source):
            return source  # type: ignore[return-value]
    return DATA_SOURCE_SPOT


def default_option_id(
    options: tuple[UnifiedViewOption, ...],
    availability: dict[str, bool],
    *,
    source: DataSourceKind | None = None,
) -> str | None:
    src = source or default_source(options, availability)
    for opt in options_for_source(options, src):
        if is_option_available(availability, opt):
            return opt.id
    return None


def option_key(source: DataSourceKind, option_id: str) -> str:
    return f"{source}:{option_id}"


def option_for(
    options: tuple[UnifiedViewOption, ...],
    option_id: str,
    *,
    source: DataSourceKind,
) -> UnifiedViewOption | None:
    for opt in options:
        if opt.id == option_id and opt.source == source:
            return opt
    return None


def option_by_id(
    options: tuple[UnifiedViewOption, ...],
    option_id: str,
) -> UnifiedViewOption | None:
    for opt in options:
        if opt.id == option_id:
            return opt
    return None
