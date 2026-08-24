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


def session_has_plan(data: SessionIcXYData) -> bool:
    if not isinstance(data, SessionIcXYData):
        return False
    if data.plan_x is None or data.plan_y is None:
        return False
    return bool(np.isfinite(data.plan_x).any() and np.isfinite(data.plan_y).any())


def any_session_has_plan(session_data: dict[str, SessionIcXYData]) -> bool:
    return any(session_has_plan(data) for data in session_data.values())
