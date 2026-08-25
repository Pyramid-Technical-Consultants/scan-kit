"""Shared data-layer types used by the source registry and view catalogs."""

from __future__ import annotations

from typing import Literal

REFERENCE_ISO = "iso"
REFERENCE_CHAMBER = "chamber"

ReferenceFrameKind = Literal["iso", "chamber"]

DATA_SOURCE_SPOT = "spot"
DATA_SOURCE_TIMESLICE = "timeslice"

DataSourceKind = Literal["spot", "timeslice"]

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
