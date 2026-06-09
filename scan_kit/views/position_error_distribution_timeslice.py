"""Position Error Distribution (Timeslice) — beam-on timeslice IC1/IC2 position-error density contours and X/Y histograms."""

from __future__ import annotations

import logging

from ..common.position_error_distribution import render_position_error_distribution
from ..common.timeslice_position_error import (
    SessionPositionErrors,
    load_session_beam_on_position_errors,
)

_log = logging.getLogger(__name__)


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Plot beam-on timeslice position error density contours and X/Y histograms by IC."""
    if not session_ids:
        print("No sessions selected")
        return

    bg_subtract = settings.bg_subtract if settings else False
    session_data: dict[str, SessionPositionErrors] = {}
    for sid in session_ids:
        errors = load_session_beam_on_position_errors(
            sid, base_dir, bg_subtract=bg_subtract
        )
        if errors is not None:
            session_data[sid] = errors

    if not session_data:
        print("No valid timeslice position error data found for any session")
        return

    render_position_error_distribution(
        session_data,
        list(session_data.keys()),
        title="Position Error Distribution (timeslice, beam-on)",
        base_dir=base_dir,
    )
