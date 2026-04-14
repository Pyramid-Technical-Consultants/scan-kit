"""Data processing utilities for scan-kit session data."""

import logging
import numpy as np
import pandas as pd

from . import transform
from . import validation

_log = logging.getLogger(__name__)
from .schema import (
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
