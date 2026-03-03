"""ZIP/CSV loading utilities for scan-kit session data."""

import zipfile

import pandas as pd


def load_csv_from_zip(zip_filename, csv_name, session_id):
    """Load a specific CSV file from a ZIP archive.

    Args:
        zip_filename: Full path to the session ZIP file (e.g., "scan_kit/{session_id}.zip").
        csv_name: Name of the CSV file (e.g., "input_map.csv", "spot_data.csv").
        session_id: Session ID used as subfolder inside the ZIP.

    Returns:
        pd.DataFrame or None if loading fails.
    """
    try:
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            with zip_ref.open(f"{session_id}/{csv_name}") as csv_file:
                return pd.read_csv(csv_file, index_col=False, skipinitialspace=True)
    except Exception as e:
        print(f"Error loading {csv_name} from session {session_id}: {e}")
        return None
