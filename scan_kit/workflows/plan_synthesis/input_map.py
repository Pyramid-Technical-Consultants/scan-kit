"""Assemble and write input_map.csv rows for plan synthesis."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

import pandas as pd

# Internal column order used while composing plans (layer_id is not exported).
INPUT_MAP_COLUMNS: tuple[str, ...] = (
    "ENERGY",
    "CURRENT",
    "BEAM_SIZE",
    "X_POSITION",
    "Y_POSITION",
    "CHARGE_REQ",
    "VELOCITY",
    "spot_no",
    "layer_id",
)

# Headers written to exported input map CSV files (benchmark / treatment-plan format).
INPUT_MAP_EXPORT_COLUMNS: tuple[str, ...] = (
    "#NO",
    "ENERGY(MeV)",
    "CURRENT(A)",
    "BEAM_SIZE(mm)",
    "X_POSITION(mm)",
    "Y_POSITION(mm)",
    "CHARGE_REQ(MU)",
    "VELOCITY(mm/s)",
)

DEFAULT_BEAM_SIZE = 3.61
DEFAULT_CURRENT_A = 1e-9


def new_layer_ids(n: int) -> list[int]:
    """Generate *n* unique random layer IDs."""
    seen: set[int] = set()
    out: list[int] = []
    while len(out) < n:
        lid = secrets.randbelow(2**31)
        if lid not in seen:
            seen.add(lid)
            out.append(lid)
    return out


def assemble_input_map(data: dict[str, list[Any]]) -> pd.DataFrame:
    """Build a DataFrame with all required columns in reference order."""
    frame = {col: data[col] for col in INPUT_MAP_COLUMNS}
    df = pd.DataFrame(frame)[list(INPUT_MAP_COLUMNS)]
    return order_input_map_by_energy(df)


def order_input_map_by_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by ENERGY descending (highest MeV first) and renumber spot_no."""
    if df.empty or "ENERGY" not in df.columns:
        return df

    ordered = df.sort_values("ENERGY", ascending=False, kind="stable").reset_index(
        drop=True
    )
    ordered = ordered.copy()
    ordered["spot_no"] = range(len(ordered))
    return ordered[list(INPUT_MAP_COLUMNS)]


def input_map_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map the internal plan frame to the benchmark input map column layout."""
    ordered = order_input_map_by_energy(df)
    return pd.DataFrame(
        {
            "#NO": ordered["spot_no"] + 1,
            "ENERGY(MeV)": ordered["ENERGY"],
            "CURRENT(A)": ordered["CURRENT"],
            "BEAM_SIZE(mm)": ordered["BEAM_SIZE"],
            "X_POSITION(mm)": ordered["X_POSITION"],
            "Y_POSITION(mm)": ordered["Y_POSITION"],
            "CHARGE_REQ(MU)": ordered["CHARGE_REQ"],
            "VELOCITY(mm/s)": ordered["VELOCITY"],
        },
        columns=list(INPUT_MAP_EXPORT_COLUMNS),
    )


def write_input_map_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write *df* as an input map CSV in benchmark treatment-plan format."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export = input_map_export_frame(df)
    export[list(INPUT_MAP_EXPORT_COLUMNS)].to_csv(
        out_path, index=False, lineterminator="\n"
    )


def plan_summary(df: pd.DataFrame) -> tuple[int, int, float]:
    """Return (n_layers, n_spots, total_mu)."""
    if df.empty:
        return 0, 0, 0.0
    n_layers = int(df["ENERGY"].nunique())
    n_spots = len(df)
    total_mu = float(df["CHARGE_REQ"].sum())
    return n_layers, n_spots, total_mu
