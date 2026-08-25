"""Adapters from canonical source payloads to Distribution Explorer types."""

from __future__ import annotations

import numpy as np

from ...common.session_ic_xy import SessionIcXYData, ic12_position_diff
from ...common.timeslice_position_error import SessionPositionErrors
from ...common.timeslice_sigma import SessionIcSigmas


def ic12_to_session_xy(payload: dict | None) -> SessionIcXYData | None:
    """Convert loaded IC1/IC2 positions into IC2−IC1 :class:`SessionIcXYData`."""
    if payload is None:
        return None
    if "ic1_x" not in payload or "ic2_x" not in payload:
        return None
    positions = SessionIcXYData(
        ic1_x=np.asarray(payload["ic1_x"], dtype=float),
        ic1_y=np.asarray(payload["ic1_y"], dtype=float),
        ic2_x=np.asarray(payload["ic2_x"], dtype=float),
        ic2_y=np.asarray(payload["ic2_y"], dtype=float),
        beam_on=payload.get("beam_on"),
    )
    return ic12_position_diff(positions)


def position_to_session_xy(payload: dict | None) -> SessionIcXYData | None:
    if payload is None:
        return None
    if "ic1_x" not in payload:
        return None
    plan_x = payload.get("plan_x")
    plan_y = payload.get("plan_y")
    return SessionIcXYData(
        ic1_x=np.asarray(payload["ic1_x"], dtype=float),
        ic1_y=np.asarray(payload["ic1_y"], dtype=float),
        ic2_x=np.asarray(payload["ic2_x"], dtype=float),
        ic2_y=np.asarray(payload["ic2_y"], dtype=float),
        plan_x=np.asarray(plan_x, dtype=float) if plan_x is not None else None,
        plan_y=np.asarray(plan_y, dtype=float) if plan_y is not None else None,
        beam_on=payload.get("beam_on"),
    )


def position_errors_to_session_errors(payload: dict | None) -> SessionPositionErrors | None:
    if payload is None:
        return None
    ic1_x = payload.get("ic1_x_err", payload.get("ic1_x"))
    ic1_y = payload.get("ic1_y_err", payload.get("ic1_y"))
    ic2_x = payload.get("ic2_x_err", payload.get("ic2_x"))
    ic2_y = payload.get("ic2_y_err", payload.get("ic2_y"))
    if ic1_x is None or ic2_x is None:
        return None
    return SessionPositionErrors(
        ic1_x=np.asarray(ic1_x, dtype=float),
        ic1_y=np.asarray(ic1_y, dtype=float),
        ic2_x=np.asarray(ic2_x, dtype=float),
        ic2_y=np.asarray(ic2_y, dtype=float),
        beam_on=payload.get("beam_on"),
    )


def sigma_to_session_sigmas(payload: dict | None) -> SessionIcSigmas | None:
    if payload is None:
        return None
    if "ic1_sig_x" not in payload:
        return None
    return SessionIcSigmas(
        ic1_x=np.asarray(payload["ic1_sig_x"], dtype=float),
        ic1_y=np.asarray(payload["ic1_sig_y"], dtype=float),
        ic2_x=np.asarray(payload["ic2_sig_x"], dtype=float),
        ic2_y=np.asarray(payload["ic2_sig_y"], dtype=float),
        beam_on=payload.get("beam_on"),
    )
