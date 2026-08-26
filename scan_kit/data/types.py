"""Shared data-layer types used by the source registry and view catalogs."""

from __future__ import annotations

from typing import Literal

REFERENCE_ISO = "iso"
REFERENCE_CHAMBER = "chamber"

ReferenceFrameKind = Literal["iso", "chamber"]

DATA_SOURCE_SPOT_ISO = "spot_iso"
DATA_SOURCE_SPOT_CHAMBER = "spot_chamber"
DATA_SOURCE_TIMESLICE_ISO = "timeslice_iso"
DATA_SOURCE_TIMESLICE_CHAMBER = "timeslice_chamber"

DataSourceKind = Literal[
    "spot_iso",
    "spot_chamber",
    "timeslice_iso",
    "timeslice_chamber",
]

ALL_DATA_SOURCES: tuple[DataSourceKind, ...] = (
    DATA_SOURCE_SPOT_ISO,
    DATA_SOURCE_SPOT_CHAMBER,
    DATA_SOURCE_TIMESLICE_ISO,
    DATA_SOURCE_TIMESLICE_CHAMBER,
)

# Coarse buckets for distribution modes (spot vs timeslice, any reference frame).
COARSE_SOURCE_SPOT = "spot"
COARSE_SOURCE_TIMESLICE = "timeslice"

CoarseDataSourceKind = Literal["spot", "timeslice"]

GRANULARITY_SPOT = "spot"
GRANULARITY_TIMESLICE_SAMPLE = "timeslice_sample"
GRANULARITY_TIMESLICE_SERIES = "timeslice_series"
GRANULARITY_LAYER = "layer"
GRANULARITY_ENERGY_BINNED = "energy_binned"
GRANULARITY_SESSION_COMPUTE = "session_compute"

GranularityKind = Literal[
    "spot",
    "timeslice_sample",
    "timeslice_series",
    "layer",
    "energy_binned",
    "session_compute",
]


def option_key(source: DataSourceKind, option_id: str) -> str:
    return f"{source}:{option_id}"


def data_source_is_timeslice(source: DataSourceKind) -> bool:
    return source.startswith("timeslice")


def coarse_data_source(source: DataSourceKind) -> CoarseDataSourceKind:
    return COARSE_SOURCE_TIMESLICE if data_source_is_timeslice(source) else COARSE_SOURCE_SPOT


def data_source_granularity(source: DataSourceKind) -> GranularityKind:
    if source.startswith("spot"):
        return GRANULARITY_SPOT
    return GRANULARITY_TIMESLICE_SAMPLE


def data_source_reference_frame(source: DataSourceKind) -> ReferenceFrameKind:
    return REFERENCE_CHAMBER if source.endswith("_chamber") else REFERENCE_ISO


def combine_data_source(
    coarse: CoarseDataSourceKind,
    reference_frame: ReferenceFrameKind,
) -> DataSourceKind:
    """Build a concrete data source from spot/timeslice and iso/chamber."""
    if coarse == COARSE_SOURCE_SPOT:
        if reference_frame == REFERENCE_CHAMBER:
            return DATA_SOURCE_SPOT_CHAMBER
        return DATA_SOURCE_SPOT_ISO
    if reference_frame == REFERENCE_CHAMBER:
        return DATA_SOURCE_TIMESLICE_CHAMBER
    return DATA_SOURCE_TIMESLICE_ISO


def split_data_source(source: DataSourceKind) -> tuple[CoarseDataSourceKind, ReferenceFrameKind]:
    return coarse_data_source(source), data_source_reference_frame(source)


def data_source_label(source: DataSourceKind) -> str:
    labels: dict[DataSourceKind, str] = {
        DATA_SOURCE_SPOT_ISO: "Spot — Isocenter",
        DATA_SOURCE_SPOT_CHAMBER: "Spot — Chamber",
        DATA_SOURCE_TIMESLICE_ISO: "Timeslice — Isocenter",
        DATA_SOURCE_TIMESLICE_CHAMBER: "Timeslice — Chamber",
    }
    return labels[source]
