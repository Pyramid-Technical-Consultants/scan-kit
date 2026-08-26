"""Timeslice confidence correlations (session compute)."""

from __future__ import annotations

import numpy as np

from ...common import detect_beam_on_mask
from ...common.session_source import load_session_timeslice_device_units, resolve_session_source
from ...common.timeslice_confidence import (
    TIMESLICE_CONFIDENCE_COLS,
    load_session_beam_on_confidence_correlations,
    resolve_timeslice_confidence_source,
)
from ..context import LoadOptions, SessionContext
from ..registry import DataSourceSpec, register
from ..types import DATA_SOURCE_TIMESLICE_ISO, GRANULARITY_TIMESLICE_SAMPLE

SOURCE_CONFIDENCE = "confidence"


def probe_confidence(ctx: SessionContext, opts: LoadOptions) -> bool:
    if opts.granularity != GRANULARITY_TIMESLICE_SAMPLE:
        return False
    src = resolve_session_source(ctx.session_id, ctx.base_dir)
    if src is None:
        return False
    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_CONFIDENCE_COLS, max_frames=1,
    )
    if not frames:
        return False
    if resolve_timeslice_confidence_source(frames[0].columns) is None:
        return False
    beam_on = detect_beam_on_mask(frames[0])
    return beam_on is not None and np.any(beam_on)


def load_confidence(ctx: SessionContext, opts: LoadOptions) -> object | None:
    if opts.granularity != GRANULARITY_TIMESLICE_SAMPLE:
        return None
    return load_session_beam_on_confidence_correlations(
        ctx.session_id,
        ctx.base_dir,
        bg_subtract=opts.resolved_bg_subtract(ctx),
    )


SPEC = register(
    DataSourceSpec(
        id=SOURCE_CONFIDENCE,
        label="Confidence Correlations",
        data_sources=frozenset({DATA_SOURCE_TIMESLICE_ISO}),
        supports_bg_subtract=True,
        supports_beam_filter=False,
        probe=probe_confidence,
        load=load_confidence,
    )
)
