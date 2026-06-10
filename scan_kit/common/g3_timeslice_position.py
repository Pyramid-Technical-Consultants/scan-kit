"""G3 timeslice IC position error (device strip frame and isocentric plan)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .schema import (
    C_LAYER_ID,
    C_SPOT_NO,
    C_X_POSITION,
    C_Y_POSITION,
    resolve_column_name,
    resolve_concept_column,
)

# G3 timeslice: fitted position minus per-IC target (same device strip frame).
_G3_POSITION_TARGET = (
    ("ic1_x", "r_ic1_x_position", "ic1_position_x_target"),
    ("ic1_y", "r_ic1_y_position", "ic1_position_y_target"),
    ("ic2_x", "r_ic2_x_position", "ic2_position_x_target"),
    ("ic2_y", "r_ic2_y_position", "ic2_position_y_target"),
)

_G3_QUALITY = (
    ("ic1_x", "ic1_x_fit_ok", "r_ic1_x_confidence", "r_ic1_x_spot_error_code"),
    ("ic1_y", "ic1_y_fit_ok", "r_ic1_y_confidence", "r_ic1_y_spot_error_code"),
    ("ic2_x", "ic2_x_fit_ok", "r_ic2_x_confidence", "r_ic2_x_spot_error_code"),
    ("ic2_y", "ic2_y_fit_ok", "r_ic2_y_confidence", "r_ic2_y_spot_error_code"),
)

_IC_SPOT_NO_CANDIDATES = {
    "ic1": ("spot_no.1", "spot_no"),
    "ic2": ("spot_no.2", "spot_no.1", "spot_no"),
}

_SPOT_SHIFT_CANDIDATES = (-2, -1, 0, 1, 2)
_STRIP_VALID = (1.0, 128.0)
G3_FIT_CONFIDENCE_MIN = 80.0
_MIN_PLAN_SPAN_MM = 0.1
_MIN_AFFINE_SLOPE = 0.01
_MAX_AFFINE_RESID_MM = 0.05
_MIN_SPOT_ANCHOR_ROWS = 10

_SPOT_DATA_ISO_ANCHOR = (
    ("ic1_x", "r_ic1_x_spot_position_raw", "r_ic1_x_spot_position"),
    ("ic1_y", "r_ic1_y_spot_position_raw", "r_ic1_y_spot_position"),
    ("ic2_x", "r_ic2_x_spot_position_raw", "r_ic2_x_spot_position"),
    ("ic2_y", "r_ic2_y_spot_position_raw", "r_ic2_y_spot_position"),
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


@dataclass(frozen=True)
class G3QualityColumns:
    ic1_x_fit_ok: str | None
    ic1_x_confidence: str | None
    ic1_x_error_code: str | None
    ic1_y_fit_ok: str | None
    ic1_y_confidence: str | None
    ic1_y_error_code: str | None
    ic2_x_fit_ok: str | None
    ic2_x_confidence: str | None
    ic2_x_error_code: str | None
    ic2_y_fit_ok: str | None
    ic2_y_confidence: str | None
    ic2_y_error_code: str | None


@dataclass(frozen=True)
class IsoAxisTransform:
    slope: float
    intercept: float


@dataclass(frozen=True)
class G3IsoTransform:
    ic1_x: IsoAxisTransform
    ic1_y: IsoAxisTransform
    ic2_x: IsoAxisTransform
    ic2_y: IsoAxisTransform
    spot_no_shift: int


@dataclass(frozen=True)
class G3IsoPlanLookup:
    """Isocentric plan targets keyed by ``(layer_id, spot_no)``."""

    layer_id: np.ndarray
    spot_no: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def lookup(
        self,
        layer_id: np.ndarray,
        spot_no: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        keys = pd.DataFrame(
            {
                "layer_id": np.asarray(layer_id, dtype=float),
                "spot_no": np.asarray(spot_no, dtype=float),
            }
        )
        table = pd.DataFrame(
            {
                "layer_id": self.layer_id.astype(float),
                "spot_no": self.spot_no.astype(float),
                "x": self.x,
                "y": self.y,
            }
        )
        merged = keys.merge(table, on=["layer_id", "spot_no"], how="left")
        return merged["x"].to_numpy(dtype=float), merged["y"].to_numpy(dtype=float)


@dataclass(frozen=True)
class G3IsoErrorContext:
    transform: G3IsoTransform
    plan: G3IsoPlanLookup
    columns: G3PositionTargetColumns
    quality: G3QualityColumns
    ic1_spot_no: str
    ic2_spot_no: str
    layer_id: str


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


def resolve_g3_quality_columns(columns) -> G3QualityColumns:
    """Resolve optional G3 fit-quality columns (all may be ``None``)."""
    resolved: dict[str, str | None] = {}
    for label, fit_ok, confidence, error_code in _G3_QUALITY:
        resolved[f"{label}_fit_ok"] = resolve_column_name(columns, fit_ok)
        resolved[f"{label}_confidence"] = resolve_column_name(columns, confidence)
        resolved[f"{label}_error_code"] = resolve_column_name(columns, error_code)
    return G3QualityColumns(**resolved)


def resolve_ic_spot_no_column(columns, ic: str) -> str | None:
    """Resolve the per-controller ``spot_no`` column for *ic* (``ic1`` / ``ic2``)."""
    for name in _IC_SPOT_NO_CANDIDATES.get(ic, ()):
        col = resolve_column_name(columns, name)
        if col is not None:
            return col
    return resolve_column_name(columns, C_SPOT_NO)


def valid_g3_fit_values(values: np.ndarray) -> np.ndarray:
    """Drop non-finite and negative samples (G3 fit failure sentinel)."""
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
    out[out < 0] = np.nan
    return out


def _fit_affine(dev: np.ndarray, iso: np.ndarray) -> tuple[float, float, float]:
    """Return ``(slope, intercept, resid_std)`` for ``iso ≈ slope*dev + intercept``."""
    ok = (
        np.isfinite(dev)
        & np.isfinite(iso)
        & (dev > _STRIP_VALID[0])
        & (dev < _STRIP_VALID[1])
    )
    if ok.sum() < 2:
        return np.nan, np.nan, np.inf
    design = np.vstack([dev[ok], np.ones(ok.sum())]).T
    slope, intercept = np.linalg.lstsq(design, iso[ok], rcond=None)[0]
    resid = iso[ok] - (slope * dev[ok] + intercept)
    return float(slope), float(intercept), float(np.std(resid))


def plan_has_position_span(plan: G3IsoPlanLookup) -> bool:
    """True when the plan map has enough X/Y spread to calibrate strip→iso."""
    x_span = float(np.nanmax(plan.x) - np.nanmin(plan.x))
    y_span = float(np.nanmax(plan.y) - np.nanmin(plan.y))
    return x_span >= _MIN_PLAN_SPAN_MM or y_span >= _MIN_PLAN_SPAN_MM


def build_g3_iso_plan_lookup(input_map: pd.DataFrame) -> G3IsoPlanLookup | None:
    """Build isocentric plan lookup from ``input_map.csv`` (flat 0,0 maps included)."""
    layer_col = resolve_concept_column(input_map.columns, C_LAYER_ID)
    spot_col = resolve_concept_column(input_map.columns, C_SPOT_NO)
    x_col = resolve_concept_column(input_map.columns, C_X_POSITION)
    y_col = resolve_concept_column(input_map.columns, C_Y_POSITION)
    if None in (layer_col, spot_col, x_col, y_col):
        return None

    table = input_map[[layer_col, spot_col, x_col, y_col]].copy()
    table.columns = ["layer_id", "spot_no", "x", "y"]
    table = table.apply(pd.to_numeric, errors="coerce").dropna()
    if table.empty:
        return None
    return G3IsoPlanLookup(
        layer_id=table["layer_id"].to_numpy(dtype=float),
        spot_no=table["spot_no"].to_numpy(dtype=float),
        x=table["x"].to_numpy(dtype=float),
        y=table["y"].to_numpy(dtype=float),
    )


def aggregate_g3_device_targets(
    frames: list[pd.DataFrame],
    cols: G3PositionTargetColumns,
) -> pd.DataFrame | None:
    """Median per-spot device-frame targets across all timeslice frames."""
    spot_col = resolve_ic_spot_no_column(frames[0].columns, "ic1")
    layer_col = resolve_column_name(frames[0].columns, C_LAYER_ID)
    if spot_col is None or layer_col is None:
        return None

    pick = [
        spot_col,
        layer_col,
        cols.ic1_x_target,
        cols.ic1_y_target,
        cols.ic2_x_target,
        cols.ic2_y_target,
    ]
    parts: list[pd.DataFrame] = []
    for df in frames:
        missing = [c for c in pick if c not in df.columns]
        if missing:
            continue
        sub = df[pick].apply(pd.to_numeric, errors="coerce")
        sub.columns = [
            "spot_no",
            "layer_id",
            "ic1_x_target",
            "ic1_y_target",
            "ic2_x_target",
            "ic2_y_target",
        ]
        parts.append(sub)
    if not parts:
        return None

    combined = pd.concat(parts, ignore_index=True)
    grouped = combined.groupby(["layer_id", "spot_no"], as_index=False).median(numeric_only=True)
    if grouped.empty:
        return None
    return grouped


def aggregate_spot_data_iso_anchor(spot_data: pd.DataFrame) -> pd.DataFrame | None:
    """Per-spot strip-frame and isocentric positions from ``spot_data.csv``."""
    layer_col = resolve_concept_column(spot_data.columns, C_LAYER_ID)
    spot_col = resolve_concept_column(spot_data.columns, C_SPOT_NO)
    if layer_col is None or spot_col is None:
        return None

    resolved: dict[str, str] = {"layer_id": layer_col, "spot_no": spot_col}
    for key, strip_name, iso_name in _SPOT_DATA_ISO_ANCHOR:
        strip_col = resolve_column_name(spot_data.columns, strip_name)
        iso_col = resolve_column_name(spot_data.columns, iso_name)
        if strip_col is None or iso_col is None:
            return None
        resolved[f"{key}_strip"] = strip_col
        resolved[f"{key}_iso"] = iso_col

    table = spot_data[list(resolved.values())].copy()
    table.columns = list(resolved.keys())
    table = table.apply(pd.to_numeric, errors="coerce").dropna()
    if len(table) < _MIN_SPOT_ANCHOR_ROWS:
        return None
    return table


def detect_spot_no_shift(
    device_targets: pd.DataFrame,
    reference: pd.DataFrame,
) -> int | None:
    """Pick the shift applied to device ``spot_no`` that best aligns with *reference*."""
    ref = reference[["layer_id", "spot_no"]].drop_duplicates()
    best_shift: int | None = None
    best_matches = -1
    for shift in _SPOT_SHIFT_CANDIDATES:
        dev = device_targets[["layer_id", "spot_no"]].drop_duplicates().copy()
        dev["spot_no"] = dev["spot_no"] + shift
        matches = len(dev.merge(ref, on=["layer_id", "spot_no"], how="inner"))
        if matches > best_matches:
            best_matches = matches
            best_shift = shift
    if best_shift is None or best_matches < _MIN_SPOT_ANCHOR_ROWS:
        return None
    return best_shift


def _axis_transform_is_valid(
    slope: float,
    intercept: float,
    resid: float,
    iso: np.ndarray,
) -> bool:
    if not np.isfinite(slope) or not np.isfinite(intercept) or not np.isfinite(resid):
        return False
    if abs(slope) < _MIN_AFFINE_SLOPE:
        return False
    iso_span = float(np.nanmax(iso) - np.nanmin(iso))
    if iso_span < _MIN_PLAN_SPAN_MM:
        return False
    return resid <= _MAX_AFFINE_RESID_MM


def _derive_axes_from_pairs(
    merged: pd.DataFrame,
    axis_specs: tuple[tuple[str, str, str], ...],
) -> dict[str, IsoAxisTransform] | None:
    axes: dict[str, IsoAxisTransform] = {}
    for key, strip_col, iso_col in axis_specs:
        slope, intercept, resid = _fit_affine(
            merged[strip_col].to_numpy(dtype=float),
            merged[iso_col].to_numpy(dtype=float),
        )
        if not _axis_transform_is_valid(
            slope,
            intercept,
            resid,
            merged[iso_col].to_numpy(dtype=float),
        ):
            return None
        axes[key] = IsoAxisTransform(slope=slope, intercept=intercept)
    return axes


def _derive_g3_iso_transform_from_plan(
    device_targets: pd.DataFrame,
    plan: G3IsoPlanLookup,
    *,
    spot_shifts: tuple[int, ...] = _SPOT_SHIFT_CANDIDATES,
) -> G3IsoTransform | None:
    """Derive per-axis strip→iso affines from commanded device vs plan targets."""
    plan_df = pd.DataFrame(
        {
            "layer_id": plan.layer_id,
            "spot_no": plan.spot_no,
            "plan_x": plan.x,
            "plan_y": plan.y,
        }
    )
    best_shift: int | None = None
    best_score = np.inf
    best_axes: dict[str, IsoAxisTransform] | None = None

    axis_specs = (
        ("ic1_x", "ic1_x_target", "plan_x"),
        ("ic1_y", "ic1_y_target", "plan_y"),
        ("ic2_x", "ic2_x_target", "plan_x"),
        ("ic2_y", "ic2_y_target", "plan_y"),
    )

    for shift in spot_shifts:
        dev = device_targets.copy()
        dev["spot_no"] = dev["spot_no"] + shift
        merged = dev.merge(plan_df, on=["layer_id", "spot_no"], how="inner")
        if len(merged) < _MIN_SPOT_ANCHOR_ROWS:
            continue

        axes = _derive_axes_from_pairs(merged, axis_specs)
        if axes is None:
            continue
        score = sum(
            _fit_affine(merged[dev_col].to_numpy(), merged[iso_col].to_numpy())[2]
            for _, dev_col, iso_col in axis_specs
        )
        if score < best_score:
            best_score = score
            best_shift = shift
            best_axes = axes

    if best_axes is None or best_shift is None:
        return None

    return G3IsoTransform(
        ic1_x=best_axes["ic1_x"],
        ic1_y=best_axes["ic1_y"],
        ic2_x=best_axes["ic2_x"],
        ic2_y=best_axes["ic2_y"],
        spot_no_shift=best_shift,
    )


def _derive_g3_iso_transform_from_spot_anchor(
    spot_anchor: pd.DataFrame,
    device_targets: pd.DataFrame,
) -> G3IsoTransform | None:
    """Derive strip→iso affines from per-spot raw/processed columns in spot_data."""
    axis_specs = (
        ("ic1_x", "ic1_x_strip", "ic1_x_iso"),
        ("ic1_y", "ic1_y_strip", "ic1_y_iso"),
        ("ic2_x", "ic2_x_strip", "ic2_x_iso"),
        ("ic2_y", "ic2_y_strip", "ic2_y_iso"),
    )
    axes = _derive_axes_from_pairs(spot_anchor, axis_specs)
    if axes is None:
        return None

    shift = detect_spot_no_shift(device_targets, spot_anchor)
    if shift is None:
        return None

    return G3IsoTransform(
        ic1_x=axes["ic1_x"],
        ic1_y=axes["ic1_y"],
        ic2_x=axes["ic2_x"],
        ic2_y=axes["ic2_y"],
        spot_no_shift=shift,
    )


def derive_g3_iso_transform(
    device_targets: pd.DataFrame,
    plan: G3IsoPlanLookup,
    spot_anchor: pd.DataFrame | None = None,
    *,
    spot_shifts: tuple[int, ...] = _SPOT_SHIFT_CANDIDATES,
) -> G3IsoTransform | None:
    """Derive strip→iso affines from plan targets, else from spot_data anchor."""
    if plan_has_position_span(plan):
        transform = _derive_g3_iso_transform_from_plan(
            device_targets, plan, spot_shifts=spot_shifts
        )
        if transform is not None:
            return transform

    if spot_anchor is not None:
        return _derive_g3_iso_transform_from_spot_anchor(spot_anchor, device_targets)
    return None


def build_g3_iso_error_context(
    input_map: pd.DataFrame,
    frames: list[pd.DataFrame],
    cols: G3PositionTargetColumns,
    spot_data: pd.DataFrame | None = None,
) -> G3IsoErrorContext | None:
    """Build isocentric error context for one G3 session."""
    plan = build_g3_iso_plan_lookup(input_map)
    if plan is None:
        return None
    device_targets = aggregate_g3_device_targets(frames, cols)
    if device_targets is None:
        return None

    spot_anchor = (
        aggregate_spot_data_iso_anchor(spot_data) if spot_data is not None else None
    )
    transform = derive_g3_iso_transform(device_targets, plan, spot_anchor)
    if transform is None:
        return None

    plan_ref = pd.DataFrame(
        {"layer_id": plan.layer_id, "spot_no": plan.spot_no}
    )
    plan_shift = detect_spot_no_shift(device_targets, plan_ref)
    if plan_shift is not None:
        transform = replace(transform, spot_no_shift=plan_shift)

    columns = frames[0].columns
    ic1_spot = resolve_ic_spot_no_column(columns, "ic1")
    ic2_spot = resolve_ic_spot_no_column(columns, "ic2")
    layer_col = resolve_column_name(columns, C_LAYER_ID)
    if ic1_spot is None or ic2_spot is None or layer_col is None:
        return None

    return G3IsoErrorContext(
        transform=transform,
        plan=plan,
        columns=cols,
        quality=resolve_g3_quality_columns(columns),
        ic1_spot_no=ic1_spot,
        ic2_spot_no=ic2_spot,
        layer_id=layer_col,
    )


def _quality_mask(
    df,
    quality: G3QualityColumns,
    axis: str,
) -> np.ndarray | None:
    fit_ok = getattr(quality, f"{axis}_fit_ok")
    confidence = getattr(quality, f"{axis}_confidence")
    error_code = getattr(quality, f"{axis}_error_code")
    if fit_ok is None and confidence is None and error_code is None:
        return None

    n = len(df)
    mask = np.ones(n, dtype=bool)
    if fit_ok is not None and fit_ok in df.columns:
        mask &= pd.to_numeric(df[fit_ok], errors="coerce").fillna(0).to_numpy() != 0
    if confidence is not None and confidence in df.columns:
        conf = pd.to_numeric(df[confidence], errors="coerce").to_numpy(dtype=float)
        mask &= np.isfinite(conf) & (conf >= G3_FIT_CONFIDENCE_MIN)
    if error_code is not None and error_code in df.columns:
        codes = pd.to_numeric(df[error_code], errors="coerce").fillna(1).to_numpy()
        mask &= codes == 0
    return mask


def _apply_quality(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return values
    out = values.copy()
    out[~mask] = np.nan
    return out


def _project(values: np.ndarray, axis: IsoAxisTransform) -> np.ndarray:
    clean = valid_g3_fit_values(values)
    return axis.slope * clean + axis.intercept


def _iso_errors_for_axis(
    df,
    meas_col: str,
    axis: IsoAxisTransform,
    plan_values: np.ndarray,
    quality: G3QualityColumns,
    axis_key: str,
) -> np.ndarray:
    meas = valid_g3_fit_values(df[meas_col].values)
    mask = _quality_mask(df, quality, axis_key)
    meas = _apply_quality(meas, mask)
    iso_meas = _project(meas, axis)
    return iso_meas - plan_values


def g3_position_error_frame_arrays(
    df,
    cols: G3PositionTargetColumns,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Position minus per-IC target for an entire timeslice frame (device strip frame)."""
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


def g3_iso_position_error_frame_arrays(
    df,
    ctx: G3IsoErrorContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Isocentric position minus stable plan target for one timeslice frame."""
    cols = ctx.columns
    transform = ctx.transform
    layer = pd.to_numeric(df[ctx.layer_id], errors="coerce").to_numpy(dtype=float)

    ic1_spot = (
        pd.to_numeric(df[ctx.ic1_spot_no], errors="coerce").to_numpy(dtype=float)
        + transform.spot_no_shift
    )
    ic2_spot = (
        pd.to_numeric(df[ctx.ic2_spot_no], errors="coerce").to_numpy(dtype=float)
        + transform.spot_no_shift
    )

    ic1_plan_x, ic1_plan_y = ctx.plan.lookup(layer, ic1_spot)
    ic2_plan_x, ic2_plan_y = ctx.plan.lookup(layer, ic2_spot)

    errors = {
        "ic1_x": _iso_errors_for_axis(
            df, cols.ic1_x, transform.ic1_x, ic1_plan_x, ctx.quality, "ic1_x"
        ),
        "ic1_y": _iso_errors_for_axis(
            df, cols.ic1_y, transform.ic1_y, ic1_plan_y, ctx.quality, "ic1_y"
        ),
        "ic2_x": _iso_errors_for_axis(
            df, cols.ic2_x, transform.ic2_x, ic2_plan_x, ctx.quality, "ic2_x"
        ),
        "ic2_y": _iso_errors_for_axis(
            df, cols.ic2_y, transform.ic2_y, ic2_plan_y, ctx.quality, "ic2_y"
        ),
    }
    if not any(np.isfinite(v).any() for v in errors.values()):
        return None
    return errors["ic1_x"], errors["ic1_y"], errors["ic2_x"], errors["ic2_y"]


def g3_position_error_arrays(
    df,
    start: int,
    end: int,
    cols: G3PositionTargetColumns,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Position minus per-IC target for one spill segment (device strip frame)."""
    frame = g3_position_error_frame_arrays(df, cols)
    if frame is None:
        return None
    return tuple(arr[start:end] for arr in frame)


def g3_iso_position_error_arrays(
    df,
    start: int,
    end: int,
    ctx: G3IsoErrorContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Isocentric position minus plan target for one spill segment."""
    frame = g3_iso_position_error_frame_arrays(df, ctx)
    if frame is None:
        return None
    return tuple(arr[start:end] for arr in frame)
