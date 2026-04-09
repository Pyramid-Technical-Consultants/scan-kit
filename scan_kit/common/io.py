"""ZIP/CSV loading utilities for scan-kit session data."""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class SessionMeta:
    """Lightweight metadata extracted from termination_summary.txt."""

    date: datetime | None
    primary_mu: float | None
    treatment_time_s: int | None

    @property
    def short_date(self) -> str:
        if self.date is None:
            return "?"
        return self.date.strftime("%m/%d/%y")

    @property
    def short_mu(self) -> str:
        if self.primary_mu is None:
            return "?"
        return f"{self.primary_mu:.1f}"

    @property
    def short_time(self) -> str:
        if self.treatment_time_s is None:
            return "?"
        return str(self.treatment_time_s)


_DATE_FMT = "%a %b %d %H:%M:%S %Y"  # e.g. "Thu Dec 11 21:36:55 2025"


def load_termination_summary(zip_path: str, session_id: str) -> SessionMeta | None:
    """Read termination_summary.txt from a session ZIP and return parsed metadata."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(f"{session_id}/termination_summary.txt") as f:
                text = f.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    date: datetime | None = None
    primary_mu: float | None = None
    treatment_s: int | None = None

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Date:"):
            try:
                date = datetime.strptime(line.split(":", 1)[1].strip(), _DATE_FMT)
            except ValueError:
                pass
        elif line.startswith("Primary total dose:"):
            try:
                primary_mu = float(line.split(":")[1].strip())
            except ValueError:
                pass
        elif line.startswith("Treatment time:"):
            try:
                treatment_s = int(float(line.split(":")[1].strip()))
            except ValueError:
                pass

    return SessionMeta(date=date, primary_mu=primary_mu, treatment_time_s=treatment_s)


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
