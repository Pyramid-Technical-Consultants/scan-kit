"""Load per-spot IC sigma measurements from session CSVs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import C_ENERGY, create_valid_mask, load_session_raw, resolve_concept_column

_log = logging.getLogger(__name__)

_SIG_KEY_VARIANTS = ("spot_sigma_raw", "spot_sigma")

IC_SIGMA_LABELS = (
    ("ic1_sig_x", "ic1", "x"),
    ("ic1_sig_y", "ic1", "y"),
    ("ic2_sig_x", "ic2", "x"),
    ("ic2_sig_y", "ic2", "y"),
)


def _resolve_sigma_col(columns, ic: str, axis: str) -> str | None:
    for key in _SIG_KEY_VARIANTS:
        for prefix in (f"r_{ic}_{axis}_{key}", f"{ic}_{axis}_{key}"):
            if prefix in columns:
                return prefix
    return None


def load_measured_sigma_spots(
    session_id: str,
    base_dir: str | Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Per-device spot arrays: ``{device: (energy_mev, sigma_mm)}``.

    Device names match ``devices.xml`` (``IC_1_X``, …). Sigma columns are scaled
    ×2 to mm (same convention as the sigma-energy view).
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
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame[create_valid_mask(frame)]
    if frame.empty:
        return None

    energies = frame[energy_col].to_numpy(dtype=float)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for device, raw_col in cols.items():
        sigmas = frame[raw_col].to_numpy(dtype=float) * 2.0
        result[device] = (energies, sigmas)
    return result


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
    for device, (energies, sigmas) in spots.items():
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
