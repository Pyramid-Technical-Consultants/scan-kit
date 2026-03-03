"""Data processing utilities for scan-kit session data."""

from pathlib import Path

import pandas as pd

from . import io
from . import transform
from . import validation


def load_session_raw(session_id, base_dir="scan_kit"):
    """Load raw input_map and spot_data for a session.

    Args:
        session_id: Session ID.
        base_dir: Base directory containing session ZIPs. Default "scan_kit".

    Returns:
        Tuple of (input_map, spot_data) DataFrames, or (None, None) if loading fails.
    """
    zip_path = Path(base_dir) / f"{session_id}.zip"
    input_map = io.load_csv_from_zip(str(zip_path), "input_map.csv", session_id)
    spot_data = io.load_csv_from_zip(str(zip_path), "spot_data.csv", session_id)
    if input_map is None or spot_data is None:
        print(f"Failed to load data for session {session_id}")
        return None, None
    return input_map, spot_data


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
        base_dir: Base directory containing session ZIPs (e.g. "scan_kit" or "test_data").
            ZIPs are expected at {base_dir}/{session_id}.zip. Default "scan_kit".

    Returns:
        Dict with session_id, ic1_x, ic1_y, ic2_x, ic2_y, energy, and any extra columns.
        Returns None if loading or validation fails.
    """
    input_map, spot_data = load_session_raw(session_id, base_dir)
    if input_map is None or spot_data is None:
        return None

    position_columns = [
        f"r_ic1_x_{position_key}",
        f"r_ic1_y_{position_key}",
        f"r_ic2_x_{position_key}",
        f"r_ic2_y_{position_key}",
    ]
    # Only add extra spot columns that exist in spot_data
    if extra_spot_columns:
        position_columns.extend(
            c for c in extra_spot_columns if c in spot_data.columns
        )
    input_columns = ["ENERGY"]
    if extra_input_columns:
        input_columns.extend(extra_input_columns)

    # Build merged dataframe (join by index to preserve row alignment)
    input_cols = [c for c in input_columns if c in input_map.columns]
    data = spot_data[position_columns].copy().join(input_map[input_cols])

    # Convert to numeric
    data = data.apply(pd.to_numeric, errors="coerce")

    # Apply validation
    valid_mask = validation.create_valid_mask(data)
    data_clean = data[valid_mask]

    if data_clean.empty:
        print(f"No valid data found for session {session_id}")
        return None

    # Apply coordinate transformations (standard 1-128 -> +/-128 mm)
    ic1_x = transform.remap(
        data_clean[f"r_ic1_x_{position_key}"], *transform.IC1_X_MAP
    )
    ic1_y = transform.remap(
        data_clean[f"r_ic1_y_{position_key}"], *transform.IC1_Y_MAP
    )
    ic2_x = transform.remap(
        data_clean[f"r_ic2_x_{position_key}"], *transform.IC2_X_MAP
    )
    ic2_y = transform.remap(
        data_clean[f"r_ic2_y_{position_key}"], *transform.IC2_Y_MAP
    )

    result = {
        "session_id": session_id,
        "ic1_x": ic1_x,
        "ic1_y": ic1_y,
        "ic2_x": ic2_x,
        "ic2_y": ic2_y,
        "energy": data_clean["ENERGY"],
    }

    # Pass through extra columns (only include columns that exist)
    for col in extra_spot_columns or []:
        if col in data_clean.columns:
            result[col] = data_clean[col].values
    for col in extra_input_columns or []:
        if col in data_clean.columns:
            result[col] = data_clean[col].values

    return result
