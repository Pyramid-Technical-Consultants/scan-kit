"""Shared catalog types for unified Qt analysis views."""

from __future__ import annotations

from dataclasses import dataclass

from ..data.types import (
    ALL_DATA_SOURCES,
    COARSE_SOURCE_SPOT,
    COARSE_SOURCE_TIMESLICE,
    CoarseDataSourceKind,
    DATA_SOURCE_SPOT_CHAMBER,
    DATA_SOURCE_SPOT_ISO,
    DATA_SOURCE_TIMESLICE_CHAMBER,
    DATA_SOURCE_TIMESLICE_ISO,
    DataSourceKind,
    data_source_label,
    option_key,
)

DATA_SOURCES: tuple[tuple[str, str], ...] = tuple(
    (source, data_source_label(source)) for source in ALL_DATA_SOURCES
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
        if source_has_available_options(options, availability, source):  # type: ignore[arg-type]
            return source  # type: ignore[return-value]
    return DATA_SOURCE_SPOT_ISO


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


PLOT_STYLE_BOX = "box"
PLOT_STYLE_VIOLIN = "violin"
PLOT_STYLE_MEAN = "mean"
PLOT_STYLE_CONTOUR = "contour"
PLOT_STYLE_SCATTER = "scatter"

BINNED_PLOT_STYLES: tuple[tuple[str, str], ...] = (
    (PLOT_STYLE_VIOLIN, "Violin"),
    (PLOT_STYLE_BOX, "Box"),
    (PLOT_STYLE_MEAN, "Mean"),
)

DISTRIBUTION_PLOT_STYLES: tuple[tuple[str, str], ...] = (
    (PLOT_STYLE_CONTOUR, "Contour"),
    (PLOT_STYLE_SCATTER, "Scatter"),
)
