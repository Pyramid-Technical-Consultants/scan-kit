"""Layer-aggregated MU delivery rate (wall-clock MU/s per energy)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import C_CHARGE_REQ, C_ENERGY, C_TIME_NS, C_TIME_S, C_TIMESTAMP, resolve_concept_column
from .session_source import load_session_csv, read_session_csv_columns, resolve_session_source

MIN_LAYER_DURATION_S = 0.05


def _spot_time_columns_available(columns: tuple[str, ...] | list[str]) -> bool:
    if "datetime" in columns:
        return True
    if resolve_concept_column(columns, C_TIME_S) is not None:
        if resolve_concept_column(columns, C_TIME_NS) is not None:
            return True
    return resolve_concept_column(columns, C_TIMESTAMP) is not None


def probe_session_mu_delivery_rates(session_id: str, base_dir: str) -> bool:
    """Cheap header-only check for layer MU delivery rate data."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return False
    input_cols = read_session_csv_columns(src, "input_map.csv")
    spot_cols = read_session_csv_columns(src, "spot_data.csv")
    if not input_cols or not spot_cols:
        return False
    if resolve_concept_column(input_cols, C_ENERGY) is None:
        return False
    if resolve_concept_column(input_cols, C_CHARGE_REQ) is None:
        return False
    return _spot_time_columns_available(spot_cols)


def _spot_wall_time_s(spot_df: pd.DataFrame) -> np.ndarray | None:
    """Return per-spot wall time in seconds (absolute or session-relative)."""
    if "datetime" in spot_df.columns:
        dt = pd.to_datetime(spot_df["datetime"], errors="coerce", utc=True)
        if dt.notna().any():
            return dt.astype("int64").to_numpy(dtype=float) / 1e9

    col_s = resolve_concept_column(spot_df.columns, C_TIME_S)
    col_ns = resolve_concept_column(spot_df.columns, C_TIME_NS)
    if col_s is not None and col_ns is not None:
        sec = pd.to_numeric(spot_df[col_s], errors="coerce").to_numpy(dtype=float)
        nsec = pd.to_numeric(spot_df[col_ns], errors="coerce").to_numpy(dtype=float)
        return sec + nsec * 1e-9

    col_ts = resolve_concept_column(spot_df.columns, C_TIMESTAMP)
    if col_ts is not None:
        return pd.to_numeric(spot_df[col_ts], errors="coerce").to_numpy(dtype=float) / 1000.0

    return None


def load_session_mu_delivery_rates(
    session_id: str,
    base_dir: str,
) -> dict[str, np.ndarray] | None:
    """Return per-energy delivery rate arrays for one session."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    input_map = load_session_csv(src, "input_map.csv")
    spot_data = load_session_csv(src, "spot_data.csv")
    if input_map is None or spot_data is None:
        return None

    col_charge = resolve_concept_column(input_map.columns, C_CHARGE_REQ)
    col_energy = resolve_concept_column(input_map.columns, C_ENERGY)
    if col_charge is None or col_energy is None:
        return None

    n = min(len(input_map), len(spot_data))
    wall_t = _spot_wall_time_s(spot_data.iloc[:n])
    if wall_t is None:
        return None

    charge = pd.to_numeric(input_map[col_charge].iloc[:n], errors="coerce").values
    ok_t = np.isfinite(wall_t)
    total_mu = float(np.nansum(charge[np.isfinite(charge)]))
    session_span_s = float(wall_t[ok_t].max() - wall_t[ok_t].min()) if ok_t.sum() >= 2 else 0.0
    if total_mu <= 0 or session_span_s < MIN_LAYER_DURATION_S:
        return None

    df = pd.DataFrame({
        "energy": pd.to_numeric(input_map[col_energy].iloc[:n], errors="coerce").values,
        "charge": charge,
        "wall_t": wall_t,
    })

    energies: list[float] = []
    rates: list[float] = []
    for energy, group in df.groupby("energy", sort=True):
        t = group["wall_t"].to_numpy(dtype=float)
        ok = np.isfinite(t)
        if ok.sum() < 2:
            continue
        duration_s = float(t[ok].max() - t[ok].min())
        if duration_s < MIN_LAYER_DURATION_S:
            continue

        mu = float(group["charge"].sum())
        if not np.isfinite(energy) or not np.isfinite(mu) or mu <= 0:
            continue

        energies.append(float(energy))
        rates.append(mu / duration_s)

    if not energies:
        return None

    order = np.argsort(energies)
    session_avg_rate = total_mu / session_span_s
    return {
        "energy": np.asarray(energies, dtype=float)[order],
        "mu_rate": np.asarray(rates, dtype=float)[order],
        "session_avg_rate": np.asarray(session_avg_rate, dtype=float),
    }
