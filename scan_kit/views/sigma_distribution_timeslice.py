"""Sigma Distribution (Timeslice) — beam-on timeslice IC1/IC2 sigma density contours and X/Y histograms."""

from __future__ import annotations

import logging

from ..common.sigma_distribution import render_sigma_distribution
from ..common.timeslice_sigma import SessionIcSigmas, load_session_beam_on_sigmas

_log = logging.getLogger(__name__)


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Plot beam-on timeslice IC sigma density contours and X/Y histograms by IC."""
    if not session_ids:
        print("No sessions selected")
        return

    bg_subtract = settings.bg_subtract if settings else False
    session_data: dict[str, SessionIcSigmas] = {}
    for sid in session_ids:
        sigmas = load_session_beam_on_sigmas(sid, base_dir, bg_subtract=bg_subtract)
        if sigmas is not None:
            session_data[sid] = sigmas

    if not session_data:
        print("No valid timeslice sigma data found for any session")
        return

    render_sigma_distribution(
        session_data,
        list(session_data.keys()),
        title="Sigma Distribution (timeslice, beam-on)",
        base_dir=base_dir,
    )
