"""ZIP/CSV loading utilities for scan-kit session data."""

import re
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


_TIMESLICE_RE = re.compile(
    r"^[^/]+/layer-(\d+)/run-(\d+)/timeslice_data_device_units\.csv$"
)


def load_timeslice_device_units(zip_filename, session_id, usecols=None):
    """Load all per-layer timeslice_data_device_units CSVs from a session ZIP.

    Discovers every ``{session_id}/layer-*/run-*/timeslice_data_device_units.csv``
    inside the archive and returns them as a list of DataFrames, each tagged with
    ``_layer_idx`` (integer parsed from the folder name).

    Args:
        zip_filename: Full path to the session ZIP.
        session_id: Session ID (subfolder inside the ZIP).
        usecols: Optional list of column names to load (passed to ``pd.read_csv``).
            Keeping this narrow is recommended — individual files can be 10+ MB.

    Returns:
        List of ``pd.DataFrame`` sorted by layer index, or an empty list on failure.
    """
    try:
        with zipfile.ZipFile(zip_filename, "r") as zf:
            matches = []
            for entry in zf.namelist():
                if not entry.startswith(f"{session_id}/"):
                    continue
                m = _TIMESLICE_RE.match(entry)
                if m:
                    matches.append((int(m.group(1)), entry))

            matches.sort(key=lambda t: t[0])
            frames = []
            for layer_idx, path in matches:
                with zf.open(path) as f:
                    df = pd.read_csv(
                        f, usecols=usecols, index_col=False, skipinitialspace=True
                    )
                df["_layer_idx"] = layer_idx
                frames.append(df)
            return frames
    except Exception as e:
        print(
            f"Error loading timeslice data from session {session_id}: {e}"
        )
        return []
