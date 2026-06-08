"""Load per-spot IC sigma measurements from session CSVs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import C_CHARGE_REQ, C_ENERGY, create_valid_mask, load_session_raw, resolve_concept_column

_log = logging.getLogger(__name__)

_SIG_KEY_VARIANTS = ("spot_sigma_raw", "spot_sigma")

IC_SIGMA_LABELS = (
    ("ic1_sig_x", "ic1", "x"),
    ("ic1_sig_y", "ic1", "y"),
    ("ic2_sig_x", "ic2", "x"),
    ("ic2_sig_y", "ic2", "y"),
)


@dataclass(frozen=True)
class MeasuredSigmaSpots:
    """Aligned per-spot sigma samples for each IC device."""

    by_device: dict[str, tuple[np.ndarray, np.ndarray]]
    weights: np.ndarray | None = None


def _resolve_sigma_col(columns, ic: str, axis: str) -> str | None:
    for key in _SIG_KEY_VARIANTS:
        for prefix in (f"r_{ic}_{axis}_{key}", f"{ic}_{axis}_{key}"):
            if prefix in columns:
                return prefix
    return None


def load_measured_sigma_spots(
    session_id: str,
    base_dir: str | Path,
) -> MeasuredSigmaSpots | None:
    """Per-device spot arrays: ``{device: (energy_mev, sigma_mm)}``.

    Device names match ``devices.xml`` (``IC_1_X``, …). Sigma columns are scaled
    ×2 to mm (same convention as the sigma-energy view).

    Optional per-spot weights come from ``charge_req`` in ``input_map.csv`` when
    present (used for dose-weighted sigma averaging).
    """
    from .devices_xml import IC_DEVICE_TO_SIG_KEY

    input_map, spot_data = load_session_raw(session_id, base_dir=base_dir)
    if input_map is None or spot_data is None:
        return None

    energy_col = resolve_concept_column(input_map.columns, C_ENERGY)
    if energy_col is None:
        _log.debug("Session %s: no energy column", session_id)
        return None

    label_to_device = {sig_key: device for device, sig_key in IC_DEVICE_TO_SIG_KEY.items()}
    cols: dict[str, str] = {}
    for label, ic, axis in IC_SIGMA_LABELS:
        col = _resolve_sigma_col(spot_data.columns, ic, axis)
        if col is not None:
            cols[label_to_device[label]] = col

    if not cols:
        _log.debug("Session %s: no sigma columns", session_id)
        return None

    frame = spot_data[list(cols.values())].copy()
    frame = frame.join(input_map[energy_col])
    weight_col = resolve_concept_column(input_map.columns, C_CHARGE_REQ)
    if weight_col is not None:
        frame = frame.join(input_map[weight_col])
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame[create_valid_mask(frame)]
    if frame.empty:
        return None

    energies = frame[energy_col].to_numpy(dtype=float)
    weights: np.ndarray | None = None
    if weight_col is not None:
        raw_weights = frame[weight_col].to_numpy(dtype=float)
        if np.any(np.isfinite(raw_weights) & (raw_weights > 0)):
            weights = raw_weights

    by_device: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for device, raw_col in cols.items():
        sigmas = frame[raw_col].to_numpy(dtype=float) * 2.0
        by_device[device] = (energies, sigmas)
    return MeasuredSigmaSpots(by_device=by_device, weights=weights)


def merge_measured_sigma_spots(parts: list[MeasuredSigmaSpots]) -> MeasuredSigmaSpots | None:
    """Concatenate per-spot samples from multiple sessions."""
    if not parts:
        return None

    by_device_lists: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    weight_parts: list[np.ndarray] = []

    for spots in parts:
        if not spots.by_device:
            continue
        ref_energies = next(iter(spots.by_device.values()))[0]
        n_spots = len(ref_energies)
        if spots.weights is not None and len(spots.weights) == n_spots:
            weight_parts.append(spots.weights)
        else:
            weight_parts.append(np.ones(n_spots, dtype=float))
        for device, (energies, sigmas) in spots.by_device.items():
            by_device_lists.setdefault(device, []).append((energies, sigmas))

    if not by_device_lists:
        return None

    merged_by_device: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for device, chunks in by_device_lists.items():
        merged_by_device[device] = (
            np.concatenate([chunk[0] for chunk in chunks]),
            np.concatenate([chunk[1] for chunk in chunks]),
        )

    weights = np.concatenate(weight_parts) if weight_parts else None
    return MeasuredSigmaSpots(by_device=merged_by_device, weights=weights)


def load_measured_sigma_spots_for_sessions(
    session_ids: list[str],
    base_dir: str | Path,
) -> tuple[MeasuredSigmaSpots | None, list[str]]:
    """Load and merge spot sigmas from every resolved *session_ids* entry."""
    warnings: list[str] = []
    parts: list[MeasuredSigmaSpots] = []
    for session_id in session_ids:
        sid = str(session_id).strip()
        if not sid:
            continue
        spots = load_measured_sigma_spots(sid, base_dir)
        if spots is None:
            warnings.append(f"Could not load sigma data for session {sid!r}.")
            continue
        parts.append(spots)

    merged = merge_measured_sigma_spots(parts)
    if merged is None:
        if not warnings:
            warnings.append("No session sigma data could be loaded.")
        return None, warnings
    return merged, warnings


def measured_sigma_by_energy(
    session_id: str,
    base_dir: str | Path,
    *,
    statistic: str = "median",
) -> dict[str, dict[float, float]] | None:
    """Aggregate spot sigmas to one value per energy layer per IC device."""
    spots = load_measured_sigma_spots(session_id, base_dir)
    if spots is None:
        return None

    agg = np.median if statistic == "median" else np.mean
    per_device: dict[str, dict[float, float]] = {}
    for device, (energies, sigmas) in spots.by_device.items():
        by_energy: dict[float, float] = {}
        for energy in np.unique(energies):
            mask = energies == energy
            if not np.any(mask):
                continue
            value = float(agg(sigmas[mask]))
            if np.isfinite(value):
                by_energy[float(energy)] = value
        if by_energy:
            per_device[device] = by_energy
    return per_device if per_device else None
