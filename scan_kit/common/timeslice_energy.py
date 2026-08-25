"""Energy lookup helpers for per-frame timeslice analysis."""

from __future__ import annotations

from . import C_ENERGY, C_LAYER_ID, resolve_concept_column
from .session_source import (
    SessionSource,
    load_session_csv,
    resolve_session_source,
)


def build_energy_lookups(input_map) -> tuple[dict | None, dict[int, float]] | None:
    """Return (layer_id → energy, layer_idx → energy) from input_map."""
    col_layer = resolve_concept_column(input_map.columns, C_LAYER_ID)
    col_energy = resolve_concept_column(input_map.columns, C_ENERGY)
    if col_energy is None:
        return None

    ordered_energies = list(dict.fromkeys(input_map[col_energy].values))
    energy_by_idx = {i: float(e) for i, e in enumerate(ordered_energies)}

    energy_by_layer: dict | None = None
    if col_layer is not None:
        energy_by_layer = input_map.groupby(col_layer)[col_energy].first().to_dict()
        if len(energy_by_layer) <= 1:
            energy_by_layer = None

    return energy_by_layer, energy_by_idx


def resolve_frame_energy(
    df,
    frame_idx: int,
    *,
    energy_by_layer: dict | None,
    energy_by_idx: dict[int, float],
    layer_col: str,
) -> float | None:
    """Resolve beam energy for one timeslice frame."""
    if df.empty:
        fallback = energy_by_idx.get(frame_idx)
        return float(fallback) if fallback is not None else None

    energy = None
    if "_layer_idx" in df.columns:
        idx = int(df["_layer_idx"].iloc[0])
        energy = energy_by_idx.get(idx)
    if energy is None and energy_by_layer is not None and layer_col in df.columns:
        lid = df[layer_col].iloc[0]
        energy = energy_by_layer.get(lid)
        if energy is None:
            try:
                energy = energy_by_layer.get(int(lid))
            except (TypeError, ValueError):
                pass
    if energy is None:
        energy = energy_by_idx.get(frame_idx)
    if energy is None:
        return None
    value = float(energy)
    return value if value == value else None  # reject NaN


def load_energy_lookups(
    session_id: str,
    base_dir: str,
) -> tuple[SessionSource, dict | None, dict[int, float]] | None:
    """Return session source and both layer_id / layer_idx energy lookups."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    input_map = load_session_csv(src, "input_map.csv")
    if input_map is None:
        return None

    lookups = build_energy_lookups(input_map)
    if lookups is None:
        return None

    energy_by_layer, energy_by_idx = lookups
    return src, energy_by_layer, energy_by_idx


def resolve_layer_col(columns) -> str | None:
    return resolve_concept_column(columns, C_LAYER_ID)
