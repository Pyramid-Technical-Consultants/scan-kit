"""Adapters from canonical source payloads to Binned Summary column dicts."""

from __future__ import annotations

import numpy as np


def ic12_to_binned_columns(payload: dict | None) -> dict | None:
    """Return energy-tagged IC2−IC1 X/Y columns, or None if energy is missing."""
    if payload is None:
        return None
    if "energy" not in payload or "ic1_x" not in payload or "ic2_x" not in payload:
        return None
    ic1_x = np.asarray(payload["ic1_x"], dtype=float)
    ic1_y = np.asarray(payload["ic1_y"], dtype=float)
    ic2_x = np.asarray(payload["ic2_x"], dtype=float)
    ic2_y = np.asarray(payload["ic2_y"], dtype=float)
    out = {
        "energy": np.asarray(payload["energy"], dtype=float),
        "ic12_x_diff": ic2_x - ic1_x,
        "ic12_y_diff": ic2_y - ic1_y,
    }
    if "session_id" in payload:
        out["session_id"] = payload["session_id"]
    if "beam_on" in payload:
        out["beam_on"] = payload["beam_on"]
    return out


def position_error_to_binned_columns(payload: dict | None) -> dict | None:
    if payload is None:
        return None
    if "energy" not in payload:
        return None
    keys = ("ic1_x_err", "ic1_y_err", "ic2_x_err", "ic2_y_err")
    if not all(k in payload for k in keys):
        return None
    out = {
        "energy": np.asarray(payload["energy"], dtype=float),
        **{k: np.asarray(payload[k], dtype=float) for k in keys},
    }
    if "plan_x" in payload:
        out["plan_x"] = np.asarray(payload["plan_x"], dtype=float)
    if "plan_y" in payload:
        out["plan_y"] = np.asarray(payload["plan_y"], dtype=float)
    if "session_id" in payload:
        out["session_id"] = payload["session_id"]
    if "beam_on" in payload:
        out["beam_on"] = payload["beam_on"]
    return out


def sigma_to_binned_columns(payload: dict | None) -> dict | None:
    if payload is None:
        return None
    if "energy" not in payload:
        return None
    keys = ("ic1_sig_x", "ic1_sig_y", "ic2_sig_x", "ic2_sig_y")
    if not any(k in payload for k in keys):
        return None
    out = {"energy": np.asarray(payload["energy"], dtype=float)}
    for k in keys:
        if k in payload:
            out[k] = np.asarray(payload[k], dtype=float)
    if "session_id" in payload:
        out["session_id"] = payload["session_id"]
    if "beam_on" in payload:
        out["beam_on"] = payload["beam_on"]
    return out
