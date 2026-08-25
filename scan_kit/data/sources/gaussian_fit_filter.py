"""Gaussian fit filter coverage (session compute)."""

from __future__ import annotations

from ...common.session_source import load_session_timeslice_device_units, resolve_session_source
from ...common.timeslice_gaussian_fit_filter_coverage import (
    TIMESLICE_GAUSSIAN_FIT_FILTER_COVERAGE_COLS,
    compute_session_gaussian_fit_filter_coverage,
    resolve_timeslice_gaussian_fit_filter_coverage_source,
)
from ..context import LoadOptions, SessionContext
from ..registry import DataSourceSpec, register
from ..types import GRANULARITY_SESSION_COMPUTE, REFERENCE_ISO

SOURCE_GAUSSIAN_FIT_FILTER = "gaussian_fit_filter"


def probe_gaussian_fit_filter(ctx: SessionContext, opts: LoadOptions) -> bool:
    if opts.granularity != GRANULARITY_SESSION_COMPUTE:
        return False
    src = resolve_session_source(ctx.session_id, ctx.base_dir)
    if src is None:
        return False
    frames = load_session_timeslice_device_units(
        src,
        usecols=TIMESLICE_GAUSSIAN_FIT_FILTER_COVERAGE_COLS,
        max_frames=1,
    )
    if not frames:
        return False
    return (
        resolve_timeslice_gaussian_fit_filter_coverage_source(frames[0].columns)
        is not None
    )


def load_gaussian_fit_filter(ctx: SessionContext, opts: LoadOptions) -> object | None:
    if opts.granularity != GRANULARITY_SESSION_COMPUTE:
        return None
    return compute_session_gaussian_fit_filter_coverage(
        ctx.session_id,
        ctx.base_dir,
        bg_subtract=opts.resolved_bg_subtract(ctx),
    )


SPEC = register(
    DataSourceSpec(
        id=SOURCE_GAUSSIAN_FIT_FILTER,
        label="Gaussian Fit Filter Coverage",
        granularities=frozenset({GRANULARITY_SESSION_COMPUTE}),
        reference_frames=frozenset({REFERENCE_ISO}),
        supports_bg_subtract=True,
        supports_beam_filter=False,
        probe=probe_gaussian_fit_filter,
        load=load_gaussian_fit_filter,
    )
)
