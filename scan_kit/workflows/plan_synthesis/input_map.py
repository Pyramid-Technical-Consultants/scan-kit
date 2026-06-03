"""Assemble and write input_map.csv rows for plan synthesis."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

import pandas as pd

# Column order matches reference session input_map.csv (including trailing comma column).
INPUT_MAP_COLUMNS: tuple[str, ...] = (
    "ENERGY",
    "CURRENT",
    "BEAM_SIZE_X",
    "BEAM_SIZE_Y",
    "X_POSITION",
    "Y_POSITION",
    "CHARGE_REQ",
    "VELOCITY",
    "spot_no",
    "layer_id",
    "beam_off",
    "map_checksum",
    "",
)

DEFAULT_BEAM_SIZE = 3.61


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
    return pd.DataFrame(frame)[list(INPUT_MAP_COLUMNS)]


def write_input_map_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write *df* as input_map.csv with reference column order."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = df[list(INPUT_MAP_COLUMNS)]
    ordered.to_csv(out_path, index=False, lineterminator="\n")


def plan_summary(df: pd.DataFrame) -> tuple[int, int, float]:
    """Return (n_layers, n_spots, total_mu)."""
    if df.empty:
        return 0, 0, 0.0
    n_layers = int(df["ENERGY"].nunique())
    n_spots = len(df)
    total_mu = float(df["CHARGE_REQ"].sum())
    return n_layers, n_spots, total_mu
