"""Confidence Correlations (Timeslice) — beam-on fit confidence vs IC current."""

from __future__ import annotations

import logging

from ..common.confidence_correlation import render_confidence_correlations
from ..common.timeslice_confidence import (
    SessionConfidenceCorrelations,
    load_session_beam_on_confidence_correlations,
)

_log = logging.getLogger(__name__)


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Plot beam-on timeslice confidence vs peak IC current and primary channel."""
    if not session_ids:
        print("No sessions selected")
        return

    bg_subtract = settings.bg_subtract if settings else False
    session_data: dict[str, SessionConfidenceCorrelations] = {}
    for sid in session_ids:
        samples = load_session_beam_on_confidence_correlations(
            sid, base_dir, bg_subtract=bg_subtract
        )
        if samples is not None:
            session_data[sid] = samples

    if not session_data:
        print("No valid timeslice confidence data found for any session")
        return

    render_confidence_correlations(
        session_data,
        list(session_data.keys()),
        title="Confidence Correlations (timeslice, beam-on)",
        base_dir=base_dir,
    )
