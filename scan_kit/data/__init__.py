"""Importing this package registers built-in sources. View modules should call
:func:`probe` / :func:`load` rather than duplicating session loaders.
"""

from __future__ import annotations

from . import sources as sources  # noqa: F401  — populate REGISTRY
from .availability import probe_session, probe_sessions
from .cache import clear_cache
from .context import LoadOptions, SessionContext
from .registry import REGISTRY, get_spec, load, probe, register
from .types import (
    ALL_DATA_SOURCES,
    COARSE_SOURCE_SPOT,
    COARSE_SOURCE_TIMESLICE,
    DATA_SOURCE_SPOT_CHAMBER,
    DATA_SOURCE_SPOT_ISO,
    DATA_SOURCE_TIMESLICE_CHAMBER,
    DATA_SOURCE_TIMESLICE_ISO,
    GRANULARITY_ENERGY_BINNED,
    GRANULARITY_LAYER,
    GRANULARITY_SESSION_COMPUTE,
    GRANULARITY_SPOT,
    GRANULARITY_TIMESLICE_SAMPLE,
    REFERENCE_CHAMBER,
    REFERENCE_ISO,
    CoarseDataSourceKind,
    DataSourceKind,
    GranularityKind,
    ReferenceFrameKind,
    coarse_data_source,
    data_source_granularity,
    data_source_is_timeslice,
    data_source_label,
    data_source_reference_frame,
    option_key,
)

__all__ = [
    "ALL_DATA_SOURCES",
    "clear_cache",
    "COARSE_SOURCE_SPOT",
    "COARSE_SOURCE_TIMESLICE",
    "DATA_SOURCE_SPOT_CHAMBER",
    "DATA_SOURCE_SPOT_ISO",
    "DATA_SOURCE_TIMESLICE_CHAMBER",
    "DATA_SOURCE_TIMESLICE_ISO",
    "GRANULARITY_ENERGY_BINNED",
    "GRANULARITY_LAYER",
    "GRANULARITY_SESSION_COMPUTE",
    "GRANULARITY_SPOT",
    "GRANULARITY_TIMESLICE_SAMPLE",
    "LoadOptions",
    "probe_session",
    "probe_sessions",
    "REFERENCE_CHAMBER",
    "REFERENCE_ISO",
    "REGISTRY",
    "option_key",
    "SessionContext",
    "CoarseDataSourceKind",
    "DataSourceKind",
    "GranularityKind",
    "ReferenceFrameKind",
    "coarse_data_source",
    "data_source_granularity",
    "data_source_is_timeslice",
    "data_source_label",
    "data_source_reference_frame",
    "get_spec",
    "load",
    "probe",
    "register",
    "sources",
]
