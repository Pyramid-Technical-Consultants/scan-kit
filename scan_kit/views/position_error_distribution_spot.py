"""Position Error Distribution (Spot) — per-spot IC1/IC2 position-error density contours and X/Y histograms.

Uses one sample per delivered spot from ``spot_data.csv`` (far fewer points than
the timeslice view). Error is the **measured** non-raw IC position (already in
plan mm coordinates) minus the **prescribed** ``X_POSITION`` / ``Y_POSITION``
from ``input_map.csv``.
"""

from __future__ import annotations

import logging

import numpy as np

from ..common import (
    C_X_POSITION,
    C_Y_POSITION,
    process_position_data,
    try_load_position_data,
)
from ..common.position_error_distribution import render_position_error_distribution
from ..common.timeslice_position_error import SessionPositionErrors

_log = logging.getLogger(__name__)


def _process_session(session_id: str, position_key: str, base_dir: str):
    """Load non-raw spot position data and compute IC1/IC2 X/Y error vs plan."""
    data = process_position_data(
        session_id,
        position_key,
        extra_input_columns=[C_X_POSITION, C_Y_POSITION],
        base_dir=base_dir,
    )
    if data is None:
        return None
    if C_X_POSITION not in data or C_Y_POSITION not in data:
        _log.debug("Session %s: input_map missing plan position columns; skipping", session_id)
        return None

    plan_x = np.asarray(data[C_X_POSITION], dtype=float)
    plan_y = np.asarray(data[C_Y_POSITION], dtype=float)
    return SessionPositionErrors(
        ic1_x=np.asarray(data["ic1_x"], dtype=float) - plan_x,
        ic1_y=np.asarray(data["ic1_y"], dtype=float) - plan_y,
        ic2_x=np.asarray(data["ic2_x"], dtype=float) - plan_x,
        ic2_y=np.asarray(data["ic2_y"], dtype=float) - plan_y,
    )


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Plot per-spot position error density contours and X/Y histograms by IC."""
    if not session_ids:
        print("No sessions selected")
        return

    session_data: dict[str, SessionPositionErrors] = {}
    for sid in session_ids:
        errors = try_load_position_data(sid, base_dir, _process_session, raw=False)
        if errors is not None:
            session_data[sid] = errors

    if not session_data:
        print("No valid spot position error data found for any session")
        return

    render_position_error_distribution(
        session_data,
        list(session_data.keys()),
        title="Position Error Distribution (spot)",
        base_dir=base_dir,
    )
