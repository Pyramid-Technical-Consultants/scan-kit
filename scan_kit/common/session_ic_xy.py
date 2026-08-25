"""Per-session IC X/Y arrays with optional planned position."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SessionIcXYData:
    ic1_x: np.ndarray
    ic1_y: np.ndarray
    ic2_x: np.ndarray
    ic2_y: np.ndarray
    plan_x: np.ndarray | None = None
    plan_y: np.ndarray | None = None
    beam_on: np.ndarray | None = None


def session_has_plan(data: SessionIcXYData) -> bool:
    if not isinstance(data, SessionIcXYData):
        return False
    if data.plan_x is None or data.plan_y is None:
        return False
    return bool(np.isfinite(data.plan_x).any() and np.isfinite(data.plan_y).any())


def any_session_has_plan(session_data: dict[str, SessionIcXYData]) -> bool:
    return any(session_has_plan(data) for data in session_data.values())


def ic12_position_diff(data: SessionIcXYData) -> SessionIcXYData:
    """IC2 minus IC1 position at each sample (X and Y)."""
    ic1_x = np.asarray(data.ic1_x, dtype=float)
    ic1_y = np.asarray(data.ic1_y, dtype=float)
    ic2_x = np.asarray(data.ic2_x, dtype=float)
    ic2_y = np.asarray(data.ic2_y, dtype=float)
    diff_x = ic2_x - ic1_x
    diff_y = ic2_y - ic1_y
    nan_x = np.full_like(diff_x, np.nan)
    nan_y = np.full_like(diff_y, np.nan)
    return SessionIcXYData(
        ic1_x=diff_x,
        ic1_y=diff_y,
        ic2_x=nan_x,
        ic2_y=nan_y,
        plan_x=None,
        plan_y=None,
        beam_on=data.beam_on,
    )
