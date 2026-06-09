"""G3 timeslice IC position error from measured position minus per-IC target."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import resolve_column_name

# G3 timeslice: fitted position minus per-IC target (same device frame).
_G3_POSITION_TARGET = (
    ("ic1_x", "r_ic1_x_position", "ic1_position_x_target"),
    ("ic1_y", "r_ic1_y_position", "ic1_position_y_target"),
    ("ic2_x", "r_ic2_x_position", "ic2_position_x_target"),
    ("ic2_y", "r_ic2_y_position", "ic2_position_y_target"),
)


@dataclass(frozen=True)
class G3PositionTargetColumns:
    ic1_x: str
    ic1_y: str
    ic2_x: str
    ic2_y: str
    ic1_x_target: str
    ic1_y_target: str
    ic2_x_target: str
    ic2_y_target: str


def resolve_g3_position_target_columns(columns) -> G3PositionTargetColumns | None:
    """Resolve G3 timeslice measured/target position column pairs."""
    resolved: dict[str, str] = {}
    for label, meas_name, tgt_name in _G3_POSITION_TARGET:
        meas_col = resolve_column_name(columns, meas_name)
        tgt_col = resolve_column_name(columns, tgt_name)
        if meas_col is None or tgt_col is None:
            return None
        resolved[label] = meas_col
        resolved[f"{label}_target"] = tgt_col
    return G3PositionTargetColumns(
        ic1_x=resolved["ic1_x"],
        ic1_y=resolved["ic1_y"],
        ic2_x=resolved["ic2_x"],
        ic2_y=resolved["ic2_y"],
        ic1_x_target=resolved["ic1_x_target"],
        ic1_y_target=resolved["ic1_y_target"],
        ic2_x_target=resolved["ic2_x_target"],
        ic2_y_target=resolved["ic2_y_target"],
    )


def valid_g3_fit_values(values: np.ndarray) -> np.ndarray:
    """Drop non-finite and negative samples (G3 fit failure sentinel)."""
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
    out[out < 0] = np.nan
    return out


def g3_position_error_frame_arrays(
    df,
    cols: G3PositionTargetColumns,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Position minus per-IC target for an entire timeslice frame."""
    pairs = (
        ("ic1_x", cols.ic1_x, cols.ic1_x_target),
        ("ic1_y", cols.ic1_y, cols.ic1_y_target),
        ("ic2_x", cols.ic2_x, cols.ic2_x_target),
        ("ic2_y", cols.ic2_y, cols.ic2_y_target),
    )
    errors: dict[str, np.ndarray] = {}
    for key, meas_col, tgt_col in pairs:
        meas = valid_g3_fit_values(df[meas_col].values)
        tgt = valid_g3_fit_values(df[tgt_col].values)
        errors[key] = meas - tgt

    if not any(np.isfinite(v).any() for v in errors.values()):
        return None
    return errors["ic1_x"], errors["ic1_y"], errors["ic2_x"], errors["ic2_y"]


def g3_position_error_arrays(
    df,
    start: int,
    end: int,
    cols: G3PositionTargetColumns,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Position minus per-IC target for one spill segment."""
    frame = g3_position_error_frame_arrays(df, cols)
    if frame is None:
        return None
    return tuple(arr[start:end] for arr in frame)
