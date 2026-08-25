"""Unified data-source registry for analysis views.

Importing this package registers built-in sources. View modules should call
:func:`probe` / :func:`load` rather than duplicating session loaders.
"""

from __future__ import annotations

from . import sources as sources  # noqa: F401  — populate REGISTRY
from .availability import probe_session, probe_sessions
from .cache import clear_cache
from .context import LoadOptions, SessionContext
from .registry import REGISTRY, get_spec, load, probe, register
from .types import (
    DATA_SOURCE_SPOT,
    DATA_SOURCE_TIMESLICE,
    GRANULARITY_ENERGY_BINNED,
    GRANULARITY_LAYER,
    GRANULARITY_SESSION_COMPUTE,
    GRANULARITY_SPOT,
    GRANULARITY_TIMESLICE_SAMPLE,
    REFERENCE_CHAMBER,
    REFERENCE_ISO,
    DataSourceKind,
    GranularityKind,
    ReferenceFrameKind,
    option_key,
)

__all__ = [
    "clear_cache",
    "DATA_SOURCE_SPOT",
    "DATA_SOURCE_TIMESLICE",
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
    "DataSourceKind",
    "GranularityKind",
    "ReferenceFrameKind",
    "get_spec",
    "load",
    "probe",
    "register",
    "sources",
]
