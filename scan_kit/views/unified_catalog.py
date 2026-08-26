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
    REFERENCE_CHAMBER,
    REFERENCE_ISO,
    ReferenceFrameKind,
    combine_data_source,
    coarse_data_source,
    data_source_label,
    data_source_reference_frame,
    option_key,
    split_data_source,
)

DATA_SOURCES: tuple[tuple[str, str], ...] = tuple(
    (source, data_source_label(source)) for source in ALL_DATA_SOURCES
)

GRANULARITY_SOURCES: tuple[tuple[CoarseDataSourceKind, str], ...] = (
    (COARSE_SOURCE_SPOT, "Spot"),
    (COARSE_SOURCE_TIMESLICE, "Timeslice"),
)

REFERENCE_FRAMES: tuple[tuple[ReferenceFrameKind, str], ...] = (
    (REFERENCE_ISO, "Isocenter"),
    (REFERENCE_CHAMBER, "Chamber"),
)

REFERENCE_FRAME_LABELS: dict[ReferenceFrameKind, str] = dict(REFERENCE_FRAMES)


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


def options_for_coarse(
    options: tuple[UnifiedViewOption, ...],
    coarse: CoarseDataSourceKind,
) -> tuple[UnifiedViewOption, ...]:
    return tuple(
        opt for opt in options
        if coarse_data_source(opt.source) == coarse
    )


def option_list_key(option: UnifiedViewOption) -> str:
    return option_key(option.source, option.id)


def option_from_list_key(
    options: tuple[UnifiedViewOption, ...],
    key: str,
) -> UnifiedViewOption | None:
    for opt in options:
        if option_list_key(opt) == key:
            return opt
    return None


def format_view_option_label(
    base_label: str,
    source: DataSourceKind,
    *,
    sibling_sources: tuple[DataSourceKind, ...],
) -> str:
    """Append an isocenter/chamber suffix when a metric has both variants."""
    coarse = coarse_data_source(source)
    ref = data_source_reference_frame(source)
    refs_for_coarse = {
        data_source_reference_frame(s)
        for s in sibling_sources
        if coarse_data_source(s) == coarse
    }
    if len(refs_for_coarse) > 1:
        return f"{base_label} ({REFERENCE_FRAME_LABELS[ref]})"
    return base_label


def registry_data_sources(source_id: str) -> tuple[DataSourceKind, ...]:
    """Sorted data sources registered for a canonical source id."""
    from ..data.registry import get_spec

    return tuple(sorted(get_spec(source_id).data_sources))


def is_option_available(
    availability: dict[str, bool],
    option: UnifiedViewOption,
) -> bool:
    composite = option_key(option.source, option.id)
    if composite in availability:
        return bool(availability[composite])
    return bool(availability.get(option.id, False))


def source_has_available_options(
    options: tuple[UnifiedViewOption, ...],
    availability: dict[str, bool],
    source: DataSourceKind,
) -> bool:
    return any(
        is_option_available(availability, opt)
        for opt in options_for_source(options, source)
    )


def coarse_has_available_options(
    options: tuple[UnifiedViewOption, ...],
    availability: dict[str, bool],
    coarse: CoarseDataSourceKind,
) -> bool:
    sources: tuple[DataSourceKind, ...] = (
        (DATA_SOURCE_SPOT_ISO, DATA_SOURCE_SPOT_CHAMBER)
        if coarse == COARSE_SOURCE_SPOT
        else (DATA_SOURCE_TIMESLICE_ISO, DATA_SOURCE_TIMESLICE_CHAMBER)
    )
    return any(
        source_has_available_options(options, availability, source)
        for source in sources
    )


def reference_frame_has_available_options(
    options: tuple[UnifiedViewOption, ...],
    availability: dict[str, bool],
    coarse: CoarseDataSourceKind,
    reference_frame: ReferenceFrameKind,
) -> bool:
    return source_has_available_options(
        options,
        availability,
        combine_data_source(coarse, reference_frame),
    )


def default_coarse_source(
    options: tuple[UnifiedViewOption, ...],
    availability: dict[str, bool],
) -> CoarseDataSourceKind:
    for coarse, _label in GRANULARITY_SOURCES:
        if coarse_has_available_options(options, availability, coarse):
            return coarse
    return COARSE_SOURCE_SPOT


def default_reference_frame(
    options: tuple[UnifiedViewOption, ...],
    availability: dict[str, bool],
    *,
    coarse: CoarseDataSourceKind,
) -> ReferenceFrameKind:
    for reference_frame, _label in REFERENCE_FRAMES:
        if reference_frame_has_available_options(
            options, availability, coarse, reference_frame,
        ):
            return reference_frame
    return REFERENCE_ISO


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
    for opt in options_for_coarse(options, coarse_data_source(src)):
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
    (PLOT_STYLE_CONTOUR, "Contour"),
    (PLOT_STYLE_SCATTER, "Scatter"),
)

DISTRIBUTION_PLOT_STYLES: tuple[tuple[str, str], ...] = (
    (PLOT_STYLE_CONTOUR, "Contour"),
    (PLOT_STYLE_SCATTER, "Scatter"),
)
