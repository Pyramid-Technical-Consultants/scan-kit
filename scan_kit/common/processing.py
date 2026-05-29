"""Data processing utilities for scan-kit session data."""

import logging
import numpy as np
import pandas as pd

from . import transform
from . import validation

_log = logging.getLogger(__name__)
from .schema import (
    C_CHARGE_REQ,
    C_ENERGY,
    C_IC1_TOTAL_DOSE,
    C_IC1_X_POS,
    C_IC1_X_POS_RAW,
    C_IC1_Y_POS,
    C_IC1_Y_POS_RAW,
    C_IC2_TOTAL_DOSE,
    C_IC2_X_POS,
    C_IC2_X_POS_RAW,
    C_IC2_Y_POS,
    C_IC2_Y_POS_RAW,
    C_IC3_TOTAL_DOSE,
    POSITION_KEY_G2,
    POSITION_KEY_G2_RAW,
    POSITION_KEY_G3,
    POSITION_KEY_G3_RAW,
    resolve_concept_column,
    resolve_requested_column,
)
from .session_source import load_session_csv, resolve_session_source


def load_session_raw(session_id, base_dir="scan_kit"):
    """Load raw input_map and spot_data for a session.

    Args:
        session_id: Session ID.
        base_dir: Base directory containing session folders or archives.

    Returns:
        Tuple of (input_map, spot_data) DataFrames, or (None, None) if loading fails.
    """
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        _log.debug("No session data found for %s", session_id)
        return None, None
    input_map = load_session_csv(src, "input_map.csv")
    spot_data = load_session_csv(src, "spot_data.csv")
    if input_map is None or spot_data is None:
        _log.debug("Failed to load CSV data for session %s", session_id)
        return None, None
    return input_map, spot_data


def try_load_position_data(session_id: str, base_dir: str, loader, *, raw: bool = True):
    """Try loading a session with G3 positions first, then G2.

    Parameters
    ----------
    raw : bool
        If True (default), use the raw register-level position keys
        (``spot_position_raw`` / ``spot_raw``).  If False, use the
        non-raw processed position keys (``spot_position`` / ``spot``),
        which are already in plan mm coordinates.
    """
    if raw:
        g3_key, g2_key = POSITION_KEY_G3_RAW, POSITION_KEY_G2_RAW
    else:
        g3_key, g2_key = POSITION_KEY_G3, POSITION_KEY_G2
    data = loader(session_id, g3_key, base_dir)
    if data is None:
        data = loader(session_id, g2_key, base_dir)
    return data


def apply_auto_calibration(
    data: dict,
    target_col: str,
    dose_cols: list[str],
) -> dict:
    """Scale each dose column so the dose-weighted mean error is zero.

    For each IC dose column, computes ``k = sum(target) / sum(delivered)``
    over finite spots where both values are positive, then multiplies the
    dose column by ``k``.  Returns a new dict (shallow copy) with the
    scaled columns.
    """
    result = dict(data)
    if target_col not in result:
        return result
    target = np.asarray(result[target_col], dtype=float)
    for col in dose_cols:
        if col not in result:
            continue
        delivered = np.asarray(result[col], dtype=float)
        ok = np.isfinite(target) & np.isfinite(delivered) & (target > 0) & (delivered > 0)
        if not ok.any():
            continue
        k = target[ok].sum() / delivered[ok].sum()
        result[col] = delivered * k
    return result


def apply_calibration_factors(
    data: dict,
    dose_cols: list[str],
    factors: dict[str, float],
) -> dict:
    """Apply pre-computed scale factors to dose columns.

    Unlike :func:`apply_auto_calibration` this does not need a target column;
    the factors are already known (e.g. computed across multiple sessions).
    """
    result = dict(data)
    for col in dose_cols:
        if col not in result or col not in factors:
            continue
        result[col] = np.asarray(result[col], dtype=float) * factors[col]
    return result


def compute_calibration_factors(
    session_ids: list[str],
    base_dir: str,
    dose_cols: list[str] | None = None,
    target_col: str | None = None,
) -> dict[str, float]:
    """Compute a single scale factor per IC across all *session_ids*.

    Pools ``target`` and ``delivered`` values from every session, then
    returns ``{col: sum(target) / sum(delivered)}`` for each dose column
    that has data.
    """
    from .schema import C_CHARGE_REQ

    if target_col is None:
        target_col = C_CHARGE_REQ
    if dose_cols is None:
        dose_cols = [C_IC1_TOTAL_DOSE, C_IC2_TOTAL_DOSE, C_IC3_TOTAL_DOSE]

    from .schema import resolve_concept_column
    from .session_source import load_session_csv, resolve_session_source

    sums_target: dict[str, float] = {c: 0.0 for c in dose_cols}
    sums_delivered: dict[str, float] = {c: 0.0 for c in dose_cols}

    for sid in session_ids:
        src = resolve_session_source(sid, base_dir)
        if src is None:
            continue
        input_map = load_session_csv(src, "input_map.csv")
        spot_data = load_session_csv(src, "spot_data.csv")
        if input_map is None or spot_data is None:
            continue

        col_target = resolve_concept_column(input_map.columns, target_col)
        if col_target is None:
            continue
        n = min(len(input_map), len(spot_data))
        target = pd.to_numeric(input_map[col_target].iloc[:n], errors="coerce").values

        for dc in dose_cols:
            col_spot = resolve_concept_column(spot_data.columns, dc)
            if col_spot is None:
                continue
            delivered = pd.to_numeric(spot_data[col_spot].iloc[:n], errors="coerce").values
            ok = (
                np.isfinite(target) & np.isfinite(delivered)
                & (target > 0) & (delivered > 0)
            )
            if not ok.any():
                continue
            sums_target[dc] += target[ok].sum()
            sums_delivered[dc] += delivered[ok].sum()

    factors: dict[str, float] = {}
    for dc in dose_cols:
        if sums_delivered[dc] > 0:
            factors[dc] = sums_target[dc] / sums_delivered[dc]
    return factors


def add_dose_ratio_columns(data: dict, *, include_ic3: bool) -> dict | None:
    """Compute ratio-difference (%) columns used by dose-ratio views."""
    required = {C_IC1_TOTAL_DOSE, C_IC2_TOTAL_DOSE}
    if not required.issubset(data):
        return None

    result = dict(data)
    ic1 = np.asarray(result[C_IC1_TOTAL_DOSE], dtype=float)
    ic2 = np.asarray(result[C_IC2_TOTAL_DOSE], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        result["ic21_ratio"] = ((ic2 / ic1) - 1.0) * 100.0

        if include_ic3 and C_IC3_TOTAL_DOSE in result:
            ic3 = np.asarray(result[C_IC3_TOTAL_DOSE], dtype=float)
            result["ic31_ratio"] = ((ic3 / ic1) - 1.0) * 100.0
            result["ic32_ratio"] = ((ic3 / ic2) - 1.0) * 100.0

    return result


DELIVERED_DOSE_COLS = {
    "ic1": C_IC1_TOTAL_DOSE,
    "ic2": C_IC2_TOTAL_DOSE,
    "ic3": C_IC3_TOTAL_DOSE,
}


def pct_error_vs_target(delivered, target) -> np.ndarray:
    """``(delivered - target) / target * 100`` where ``|target| > 0``; else NaN."""
    d = np.asarray(delivered, dtype=float)
    t = np.asarray(target, dtype=float)
    out = np.full_like(d, np.nan, dtype=float)
    ok = np.isfinite(d) & np.isfinite(t) & (np.abs(t) > 1e-15)
    out[ok] = (d[ok] - t[ok]) / t[ok] * 100.0
    return out


def add_dose_error_columns(data: dict, *, target_col: str = C_CHARGE_REQ,
                           delivered_cols: dict | None = None) -> dict | None:
    """Add per-spot ``{ic}_dose_err_pct`` (% of target) columns for each IC present.

    Shared by the dose-error views. Returns a new dict, or ``None`` when the
    target column or every delivered-dose column is missing.
    """
    delivered_cols = delivered_cols or DELIVERED_DOSE_COLS
    if target_col not in data:
        return None
    if not any(col in data for col in delivered_cols.values()):
        return None

    result = dict(data)
    target = result[target_col]
    for ic, col in delivered_cols.items():
        if col in result:
            result[f"{ic}_dose_err_pct"] = pct_error_vs_target(result[col], target)
    return result


def filter_data_rows(data: dict, keep_mask, *, skip_keys=("session_id",)) -> dict:
    """Filter all array/Series-like values in ``data`` with a boolean mask."""
    keep = np.asarray(keep_mask, dtype=bool)
    idx = np.flatnonzero(keep)
    filtered = dict(data)

    for key, val in data.items():
        if key in skip_keys:
            continue
        if isinstance(val, np.ndarray):
            filtered[key] = val[keep]
        elif hasattr(val, "iloc"):
            filtered[key] = val.iloc[idx]

    return filtered


def add_spot_delivery_time(data: dict, *, max_spot_time_ms: float = 100.0) -> dict | None:
    """Add ``spot_time`` and filter rows above ``max_spot_time_ms``."""
    if "timestamp" not in data or "layer_id" not in data:
        return None

    result = dict(data)
    df = pd.DataFrame(
        {
            "timestamp": np.asarray(result["timestamp"], dtype=float),
            "layer_id": np.asarray(result["layer_id"]),
        }
    )
    spot_time = df.groupby("layer_id")["timestamp"].diff()
    first_mask = spot_time.isna()
    spot_time.loc[first_mask] = df.loc[first_mask, "timestamp"]
    st = spot_time.to_numpy()

    keep = np.isfinite(st) & (st <= max_spot_time_ms)
    result = filter_data_rows(result, keep)
    result["spot_time"] = st[keep]
    return result


def process_position_data(
    session_id,
    position_key,
    extra_spot_columns=None,
    extra_input_columns=None,
    base_dir="scan_kit",
):
    """Process session data and return cleaned position data.

    Loads input_map and spot_data, validates, applies coordinate remap,
    and returns a dict with standard position fields plus any extra columns.

    Args:
        session_id: Session ID.
        position_key: Column key for position data (e.g., "spot_position_raw", "spot_raw").
        extra_spot_columns: Optional list of extra column names from spot_data to include.
        extra_input_columns: Optional list of extra column names from input_map to include.
        base_dir: Directory containing session data: unpacked ``{session_id}/`` folders,
            ``{session_id}.zip``, or ``{session_id}.tgz`` / ``.tar.gz`` / etc.
            Default "scan_kit".

    Returns:
        Dict with session_id, ic1_x, ic1_y, ic2_x, ic2_y, energy, and any extra columns.
        Returns None if loading or validation fails.
    """
    input_map, spot_data = load_session_raw(session_id, base_dir)
    if input_map is None or spot_data is None:
        return None

    is_raw = position_key.endswith("_raw")

    if is_raw:
        required_position_concepts = {
            C_IC1_X_POS_RAW: transform.IC1_X_MAP,
            C_IC1_Y_POS_RAW: transform.IC1_Y_MAP,
            C_IC2_X_POS_RAW: transform.IC2_X_MAP,
            C_IC2_Y_POS_RAW: transform.IC2_Y_MAP,
        }
    else:
        required_position_concepts = {
            C_IC1_X_POS: None,
            C_IC1_Y_POS: None,
            C_IC2_X_POS: None,
            C_IC2_Y_POS: None,
        }

    resolved_position: dict[str, str] = {}
    missing_required = []
    for concept in required_position_concepts:
        resolved = resolve_concept_column(
            spot_data.columns, concept, position_key=position_key
        )
        if resolved is None:
            missing_required.append(concept)
            continue
        resolved_position[concept] = resolved
    if missing_required:
        _log.debug("Session %s: missing position concepts %s", session_id, missing_required)
        return None

    spot_columns = list(resolved_position.values())
    resolved_extra_spot: dict[str, str] = {}
    if extra_spot_columns:
        for req in extra_spot_columns:
            resolved = resolve_concept_column(spot_data.columns, req)
            if resolved is None:
                resolved = resolve_requested_column(spot_data.columns, req)
            if resolved is None:
                continue
            resolved_extra_spot[req] = resolved
            if resolved not in spot_columns:
                spot_columns.append(resolved)

    input_requests = [C_ENERGY]
    if extra_input_columns:
        input_requests.extend(extra_input_columns)

    resolved_input: dict[str, str] = {}
    for req in input_requests:
        resolved = resolve_concept_column(input_map.columns, req)
        if resolved is None:
            resolved = resolve_requested_column(input_map.columns, req)
        if resolved is not None:
            resolved_input[req] = resolved
    if C_ENERGY not in resolved_input:
        _log.debug("Session %s: missing ENERGY column", session_id)
        return None

    # Build merged dataframe (join by index to preserve row alignment)
    input_cols = list(dict.fromkeys(resolved_input.values()))
    data = spot_data[spot_columns].copy().join(input_map[input_cols])

    # Convert to numeric
    data = data.apply(pd.to_numeric, errors="coerce")

    # Apply validation
    valid_mask = validation.create_valid_mask(data)
    data_clean = data[valid_mask]

    if data_clean.empty:
        _log.debug("No valid data after validation for session %s", session_id)
        return None

    if is_raw:
        ic1_x = transform.remap(
            data_clean[resolved_position[C_IC1_X_POS_RAW]], *transform.IC1_X_MAP
        )
        ic1_y = transform.remap(
            data_clean[resolved_position[C_IC1_Y_POS_RAW]], *transform.IC1_Y_MAP
        )
        ic2_x = transform.remap(
            data_clean[resolved_position[C_IC2_X_POS_RAW]], *transform.IC2_X_MAP
        )
        ic2_y = transform.remap(
            data_clean[resolved_position[C_IC2_Y_POS_RAW]], *transform.IC2_Y_MAP
        )
    else:
        ic1_x = pd.to_numeric(data_clean[resolved_position[C_IC1_X_POS]], errors="coerce")
        ic1_y = pd.to_numeric(data_clean[resolved_position[C_IC1_Y_POS]], errors="coerce")
        ic2_x = pd.to_numeric(data_clean[resolved_position[C_IC2_X_POS]], errors="coerce")
        ic2_y = pd.to_numeric(data_clean[resolved_position[C_IC2_Y_POS]], errors="coerce")

    result = {
        "session_id": session_id,
        "ic1_x": ic1_x,
        "ic1_y": ic1_y,
        "ic2_x": ic2_x,
        "ic2_y": ic2_y,
        "energy": data_clean[resolved_input[C_ENERGY]],
    }

    # Pass through extra columns (only include columns that exist)
    for col in extra_spot_columns or []:
        resolved = resolved_extra_spot.get(col)
        if resolved in data_clean.columns:
            result[col] = data_clean[resolved].values
    for col in extra_input_columns or []:
        resolved = resolved_input.get(col)
        if resolved in data_clean.columns:
            result[col] = data_clean[resolved].values

    return result


# ---------------------------------------------------------------------------
# Timeslice signal helpers
# ---------------------------------------------------------------------------

BG_ROLLING_WINDOW = 200
DEFAULT_THRESHOLD_FRAC = 0.10


def sliding_background(
    signal: np.ndarray,
    threshold_frac: float = DEFAULT_THRESHOLD_FRAC,
    rolling_window: int = BG_ROLLING_WINDOW,
    beam_off_mask: np.ndarray | None = None,
):
    """Compute a per-sample background that tracks drift within a layer.

    Parameters
    ----------
    beam_off_mask : optional bool array
        When provided, only samples where this mask is True **and** the
        current-threshold condition is met are used for background
        estimation.  This prevents beam-on residual current from leaking
        into the background model.

    Returns ``(bg_array, bg_global, peak)`` where *bg_array* has the same
    length as *signal*.  If the layer is too short or flat, *bg_array* is
    a constant filled with *bg_global*.
    """
    from scipy.ndimage import median_filter

    low_mask = signal <= np.nanpercentile(signal, 25)
    bg_global = float(
        np.nanmedian(signal[low_mask]) if low_mask.any() else np.nanpercentile(signal, 25)
    )
    peak = float(np.nanpercentile(signal, 99))
    if peak - bg_global < 1.0:
        return np.full(len(signal), bg_global), bg_global, peak

    thresh = threshold_frac * (peak - bg_global)
    beam_off = (signal - bg_global) <= thresh
    if beam_off_mask is not None:
        beam_off = beam_off & beam_off_mask

    off_idx = np.where(beam_off)[0]
    if len(off_idx) >= rolling_window:
        smoothed = median_filter(signal[off_idx], size=rolling_window, mode="reflect")
        bg_array = np.interp(np.arange(len(signal)), off_idx, smoothed)
    else:
        bg_array = np.full(len(signal), bg_global)

    return bg_array, bg_global, peak


_IC_CURRENT_CONCEPTS = [
    "ic1_current",
    "ic2_current",
    "ic1_strip_sum",
    "ic2_strip_sum",
    "ic3_current_a",
    "ic3_current_b",
    "ic3_current_c",
    "ic3_current_d",
]


def _detect_beam_off_mask(df, *, strict: bool = False) -> np.ndarray | None:
    """Build a boolean mask that is True only when the beam is confirmed off.

    With ``strict=False`` (default) only the spill-level gate is used:
        G3: ``rci_in_trigger == 0``
        G2: ``r_beamOk == 0``

    With ``strict=True`` the extraction/enable signal is AND-ed in, giving
    interspill (spot-level) masking:
        G3: ``rci_in_trigger == 0  OR  rci_out_kicker == 0``
        G2: ``r_beamOk == 0  OR  r_beamEnabled == 0``
    """
    if "rci_in_trigger" in df.columns:
        off = df["rci_in_trigger"].values == 0
        if strict and "rci_out_kicker" in df.columns:
            off = off | (df["rci_out_kicker"].values == 0)
        return off
    if "r_beamOk" in df.columns:
        off = df["r_beamOk"].values == 0
        if strict and "r_beamEnabled" in df.columns:
            off = off | (df["r_beamEnabled"].values == 0)
        return off
    return None


def subtract_background_frames(frames: list) -> list:
    """Apply sliding background subtraction to IC current columns in-place.

    *frames* is a list of DataFrames as returned by
    ``load_session_timeslice_device_units``.  Each IC current column is
    independently background-subtracted using :func:`sliding_background`.
    A hardware beam-off signal (G3 ``rci_in_trigger``, G2 ``r_beamOk``)
    is used when available to restrict which samples inform the background.
    Returns the same list (mutated) for convenience.
    """
    for df in frames:
        beam_off = _detect_beam_off_mask(df)
        for col in _IC_CURRENT_CONCEPTS:
            if col not in df.columns:
                continue
            raw = df[col].values.astype(float)
            bg, _, _ = sliding_background(raw, beam_off_mask=beam_off)
            df[col] = raw - bg
    return frames
