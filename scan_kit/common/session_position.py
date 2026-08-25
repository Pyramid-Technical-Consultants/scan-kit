"""Load per-spot or timeslice IC position errors from session data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from . import (
    C_CHARGE_REQ,
    C_LAYER_ID,
    C_X_POSITION,
    C_Y_POSITION,
    detect_beam_on_mask,
    process_position_data,
    resolve_concept_column,
    try_load_position_data,
)
from .devices_xml import IC_SIGMA_DEVICES
from .session_source import load_session_csv, load_session_timeslice_device_units, resolve_session_source
from .timeslice_energy import build_energy_lookups, resolve_frame_energy
from .timeslice_position_error import (
    TIMESLICE_POSITION_ERROR_COLS,
    frame_timeslice_error_arrays,
    resolve_session_timeslice_error_source,
)

_log = logging.getLogger(__name__)

PositionDataSource = Literal["spot", "timeslice"]
DEFAULT_POSITION_DATA_SOURCE: PositionDataSource = "spot"

_DEVICE_ERROR_FIELDS = {
    "IC_1_X": "ic1_x_err",
    "IC_1_Y": "ic1_y_err",
    "IC_2_X": "ic2_x_err",
    "IC_2_Y": "ic2_y_err",
}


@dataclass(frozen=True)
class MeasuredPositionErrors:
    """Aligned position-error samples for each IC device."""

    by_device: dict[str, tuple[np.ndarray, np.ndarray]]
    weights: np.ndarray | None = None


def normalize_position_data_source(value: str | None) -> PositionDataSource:
    if value == "timeslice":
        return "timeslice"
    return DEFAULT_POSITION_DATA_SOURCE


def _spot_position_loader(session_id: str, position_key: str, base_dir: str):
    return process_position_data(
        session_id,
        position_key,
        extra_input_columns=[C_X_POSITION, C_Y_POSITION, C_CHARGE_REQ],
        base_dir=base_dir,
    )


def _errors_from_spot_data(data: dict) -> MeasuredPositionErrors | None:
    if C_X_POSITION not in data or C_Y_POSITION not in data:
        return None

    plan_x = np.asarray(data[C_X_POSITION], dtype=float)
    plan_y = np.asarray(data[C_Y_POSITION], dtype=float)
    ic1_x = np.asarray(data["ic1_x"], dtype=float)
    ic1_y = np.asarray(data["ic1_y"], dtype=float)
    ic2_x = np.asarray(data["ic2_x"], dtype=float)
    ic2_y = np.asarray(data["ic2_y"], dtype=float)

    energies = np.asarray(data["energy"], dtype=float)
    errors_by_field = {
        "ic1_x_err": ic1_x - plan_x,
        "ic1_y_err": ic1_y - plan_y,
        "ic2_x_err": ic2_x - plan_x,
        "ic2_y_err": ic2_y - plan_y,
    }

    weights: np.ndarray | None = None
    if C_CHARGE_REQ in data:
        raw_weights = np.asarray(data[C_CHARGE_REQ], dtype=float)
        if np.any(np.isfinite(raw_weights) & (raw_weights > 0)):
            weights = raw_weights

    by_device: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for device in IC_SIGMA_DEVICES:
        field = _DEVICE_ERROR_FIELDS[device]
        by_device[device] = (energies, errors_by_field[field])

    return MeasuredPositionErrors(by_device=by_device, weights=weights)


def load_measured_position_errors_spot(
    session_id: str,
    base_dir: str | Path,
) -> MeasuredPositionErrors | None:
    """Per-device spot arrays: ``{device: (energy_mev, error_mm)}``."""
    data = try_load_position_data(
        session_id,
        str(base_dir),
        _spot_position_loader,
        raw=False,
    )
    if data is None:
        _log.debug("Session %s: no processed spot position data", session_id)
        return None
    return _errors_from_spot_data(data)


def load_measured_position_errors_timeslice(
    session_id: str,
    base_dir: str | Path,
) -> MeasuredPositionErrors | None:
    """Per-device timeslice arrays: ``{device: (energy_mev, error_mm)}``."""
    src = resolve_session_source(session_id, str(base_dir))
    if src is None:
        return None

    input_map = load_session_csv(src, "input_map.csv")
    if input_map is None:
        return None

    lookups = build_energy_lookups(input_map)
    if lookups is None:
        return None
    energy_by_layer, energy_by_idx = lookups

    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS
    )
    if not frames:
        return None

    error_source = resolve_session_timeslice_error_source(src, frames)
    if error_source is None:
        return None

    layer_col = resolve_concept_column(frames[0].columns, C_LAYER_ID)

    ic1_x_parts: list[np.ndarray] = []
    ic1_y_parts: list[np.ndarray] = []
    ic2_x_parts: list[np.ndarray] = []
    ic2_y_parts: list[np.ndarray] = []
    energy_parts: list[np.ndarray] = []

    for frame_idx, df in enumerate(frames):
        beam_on = detect_beam_on_mask(df)
        if beam_on is None:
            continue

        energy = resolve_frame_energy(
            df,
            frame_idx,
            energy_by_layer=energy_by_layer,
            energy_by_idx=energy_by_idx,
            layer_col=layer_col or "",
        )
        if energy is None:
            continue

        frame_errors = frame_timeslice_error_arrays(df, error_source)
        if frame_errors is None:
            continue

        ic1_x, ic1_y, ic2_x, ic2_y = frame_errors
        ic1_x = ic1_x[beam_on]
        ic1_y = ic1_y[beam_on]
        ic2_x = ic2_x[beam_on]
        ic2_y = ic2_y[beam_on]
        if not any(np.isfinite(arr).any() for arr in (ic1_x, ic1_y, ic2_x, ic2_y)):
            continue

        n = len(ic1_x)
        ic1_x_parts.append(ic1_x)
        ic1_y_parts.append(ic1_y)
        ic2_x_parts.append(ic2_x)
        ic2_y_parts.append(ic2_y)
        energy_parts.append(np.full(n, energy, dtype=float))

    if not ic1_x_parts:
        return None

    energies = np.concatenate(energy_parts)
    by_device = {
        "IC_1_X": (energies, np.concatenate(ic1_x_parts)),
        "IC_1_Y": (energies, np.concatenate(ic1_y_parts)),
        "IC_2_X": (energies, np.concatenate(ic2_x_parts)),
        "IC_2_Y": (energies, np.concatenate(ic2_y_parts)),
    }
    return MeasuredPositionErrors(by_device=by_device, weights=None)


def merge_measured_position_errors(
    parts: list[MeasuredPositionErrors],
) -> MeasuredPositionErrors | None:
    """Concatenate position-error samples from multiple sessions."""
    if not parts:
        return None

    by_device_lists: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    weight_parts: list[np.ndarray] = []

    for measured in parts:
        if not measured.by_device:
            continue
        ref_energies = next(iter(measured.by_device.values()))[0]
        n_samples = len(ref_energies)
        if measured.weights is not None and len(measured.weights) == n_samples:
            weight_parts.append(measured.weights)
        else:
            weight_parts.append(np.ones(n_samples, dtype=float))
        for device, arrays in measured.by_device.items():
            by_device_lists.setdefault(device, []).append(arrays)

    if not by_device_lists:
        return None

    merged_by_device: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for device, chunks in by_device_lists.items():
        merged_by_device[device] = (
            np.concatenate([chunk[0] for chunk in chunks]),
            np.concatenate([chunk[1] for chunk in chunks]),
        )

    weights = np.concatenate(weight_parts) if weight_parts else None
    return MeasuredPositionErrors(by_device=merged_by_device, weights=weights)


def load_measured_position_errors_for_sessions(
    session_ids: list[str],
    base_dir: str | Path,
    *,
    data_source: PositionDataSource = DEFAULT_POSITION_DATA_SOURCE,
) -> tuple[MeasuredPositionErrors | None, list[str]]:
    """Load and merge position errors from every resolved *session_ids* entry."""
    loader = (
        load_measured_position_errors_spot
        if data_source == "spot"
        else load_measured_position_errors_timeslice
    )
    warnings: list[str] = []
    parts: list[MeasuredPositionErrors] = []
    for session_id in session_ids:
        sid = str(session_id).strip()
        if not sid:
            continue
        measured = loader(sid, base_dir)
        if measured is None:
            warnings.append(f"Could not load position data for session {sid!r}.")
            continue
        parts.append(measured)

    merged = merge_measured_position_errors(parts)
    if merged is None:
        if not warnings:
            warnings.append("No session position data could be loaded.")
        return None, warnings
    return merged, warnings
