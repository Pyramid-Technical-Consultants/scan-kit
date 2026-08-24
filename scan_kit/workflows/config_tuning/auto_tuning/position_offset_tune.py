"""Apply measured session position errors to ``devices.xml`` zero offsets."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from scan_kit.common.devices_xml import IC_SIGMA_DEVICES
from scan_kit.common.session_position import (
    DEFAULT_POSITION_DATA_SOURCE,
    MeasuredPositionErrors,
    PositionDataSource,
    load_measured_position_errors_for_sessions,
)

from .sigma_tune import (
    DEFAULT_SIGMA_OPTIMIZE_MODE,
    SigmaOptimizeMode,
    compute_band_sigma,
    format_sigma_k0,
)

PositionOptimizeMode = SigmaOptimizeMode


def format_offset_mm(value: float) -> str:
    """Format offset values the way room ``devices.xml`` files typically store them."""
    return format_sigma_k0(value)


@dataclass(frozen=True)
class PositionOffsetTunePreviewRow:
    """One energy band and its proposed zero-offset change for one IC device."""

    device: str
    min_energy: float
    max_energy: float
    old_offset: float
    new_offset: float
    n_samples: int
    error_variance: float
    residual_variance: float
    max_abs_error_mm: float
    max_residual_mm: float
    band_median_error: float

    @property
    def energy_center_mev(self) -> float:
        return (self.min_energy + self.max_energy) / 2.0

    @property
    def delta_offset(self) -> float:
        return self.new_offset - self.old_offset


@dataclass
class PositionOffsetTuneResult:
    offsets_updated: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.offsets_updated > 0


@dataclass
class _OffsetUpdate:
    element: ET.Element
    device: str
    old_offset: float
    new_offset: float
    correction: float


_DEVICE_ORDER = {name: index for index, name in enumerate(IC_SIGMA_DEVICES)}


def read_zero_offsets_from_tree(
    root: ET.Element,
    *,
    devices: tuple[str, ...] = IC_SIGMA_DEVICES,
) -> dict[str, tuple[ET.Element, float]]:
    """Return ``{device: (zero_offset_at_iso_mm element, value)}``."""
    device_set = set(devices)
    found: dict[str, tuple[ET.Element, float]] = {}
    for chamber in root.iter("ion_chamber"):
        device_el = chamber.find("device")
        if device_el is None:
            continue
        name = device_el.get("name")
        if not name or name not in device_set:
            continue
        offset_el = chamber.find("zero_offset_at_iso_mm")
        if offset_el is None or offset_el.text is None:
            continue
        try:
            value = float(offset_el.text.strip())
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            found[name] = (offset_el, value)
    return found


def compute_global_offset_correction(
    errors: np.ndarray,
    weights: np.ndarray | None,
    mode: PositionOptimizeMode,
) -> float:
    """Reduce all samples for one axis to a single correction (mm)."""
    finite = np.isfinite(errors)
    if not np.any(finite):
        return float("nan")
    valid_errors = errors[finite]
    valid_weights = weights[finite] if weights is not None else None
    return compute_band_sigma(valid_errors, valid_weights, mode)


def band_error_variance(errors: np.ndarray) -> float:
    """Sample variance (mm²) of position errors in one energy band."""
    finite = errors[np.isfinite(errors)]
    if finite.size < 2:
        return 0.0
    return float(np.var(finite, ddof=1))


def band_max_abs_error_mm(errors: np.ndarray) -> float:
    """Largest absolute position error in one energy band."""
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(np.abs(finite)))


def band_max_residual_mm(residuals: np.ndarray) -> float:
    """Largest absolute residual in one energy band."""
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(np.abs(finite)))


def collect_offset_updates(
    root: ET.Element,
    measured: MeasuredPositionErrors,
    *,
    devices: tuple[str, ...] = IC_SIGMA_DEVICES,
    optimize_mode: PositionOptimizeMode = DEFAULT_SIGMA_OPTIMIZE_MODE,
) -> tuple[list[_OffsetUpdate], list[str]]:
    """Collect per-device offset updates without mutating *root*."""
    updates: list[_OffsetUpdate] = []
    warnings: list[str] = []
    offsets = read_zero_offsets_from_tree(root, devices=devices)

    for device in devices:
        offset_entry = offsets.get(device)
        spot_data = measured.by_device.get(device)
        if offset_entry is None:
            warnings.append(f"No zero_offset_at_iso_mm found for {device}.")
            continue
        if spot_data is None:
            warnings.append(f"No session position data for {device}.")
            continue

        _, errors = spot_data
        correction = compute_global_offset_correction(
            errors,
            measured.weights,
            optimize_mode,
        )
        if not np.isfinite(correction):
            warnings.append(f"Could not compute offset correction for {device}.")
            continue

        element, old_offset = offset_entry
        new_offset = old_offset - correction
        updates.append(
            _OffsetUpdate(
                element=element,
                device=device,
                old_offset=old_offset,
                new_offset=new_offset,
                correction=correction,
            )
        )

    if not updates and not warnings:
        warnings.append("No zero_offset_at_iso_mm elements matched session data.")
    return updates, warnings


def _corrections_by_device(updates: list[_OffsetUpdate]) -> dict[str, float]:
    return {update.device: update.correction for update in updates}


def collect_band_preview_rows(
    root: ET.Element,
    measured: MeasuredPositionErrors,
    updates: list[_OffsetUpdate],
) -> list[PositionOffsetTunePreviewRow]:
    """Build per-energy-band preview rows using global offset changes."""
    corrections = _corrections_by_device(updates)
    new_offsets = {u.device: u.new_offset for u in updates}
    old_offsets = {u.device: u.old_offset for u in updates}
    rows: list[PositionOffsetTunePreviewRow] = []

    seen_bands: set[tuple[str, float, float]] = set()
    for chamber in root.iter("ion_chamber"):
        device_el = chamber.find("device")
        if device_el is None:
            continue
        device = device_el.get("name")
        if not device or device not in corrections:
            continue

        spot_data = measured.by_device.get(device)
        if spot_data is None:
            continue
        energies, errors = spot_data
        correction = corrections[device]

        for el in chamber.findall("beam_sigma_conversions"):
            if el.get("in_units", "").upper() != "MEV":
                continue
            if el.get("out_units", "").lower() != "mm":
                continue
            try:
                min_e = float(el.get("min_energy", "nan"))
                max_e = float(el.get("max_energy", "nan"))
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(min_e) and np.isfinite(max_e)):
                continue

            band_key = (device, min_e, max_e)
            if band_key in seen_bands:
                continue
            seen_bands.add(band_key)

            mask = (energies >= min_e) & (energies <= max_e)
            if not np.any(mask):
                continue

            band_errors = errors[mask]
            finite = band_errors[np.isfinite(band_errors)]
            if finite.size == 0:
                continue

            residuals = band_errors - correction
            rows.append(
                PositionOffsetTunePreviewRow(
                    device=device,
                    min_energy=min_e,
                    max_energy=max_e,
                    old_offset=old_offsets[device],
                    new_offset=new_offsets[device],
                    n_samples=int(np.count_nonzero(mask)),
                    error_variance=band_error_variance(band_errors),
                    residual_variance=band_error_variance(residuals),
                    max_abs_error_mm=band_max_abs_error_mm(band_errors),
                    max_residual_mm=band_max_residual_mm(residuals),
                    band_median_error=float(np.median(finite)),
                )
            )

    rows.sort(
        key=lambda row: (
            _DEVICE_ORDER.get(row.device, 99),
            -row.energy_center_mev,
        )
    )
    return rows


def compute_position_offset_tune_preview(
    root: ET.Element,
    session_ids: list[str],
    base_dir: str,
    *,
    data_source: PositionDataSource = DEFAULT_POSITION_DATA_SOURCE,
    optimize_mode: PositionOptimizeMode = DEFAULT_SIGMA_OPTIMIZE_MODE,
) -> tuple[list[PositionOffsetTunePreviewRow], list[str]]:
    """Return proposed zero offsets for every matching band in *root*."""
    measured, load_warnings = load_measured_position_errors_for_sessions(
        session_ids,
        base_dir,
        data_source=data_source,
    )
    if measured is None:
        return [], load_warnings
    updates, warnings = collect_offset_updates(
        root,
        measured,
        optimize_mode=optimize_mode,
    )
    rows = collect_band_preview_rows(root, measured, updates)
    return rows, load_warnings + warnings


def apply_position_offsets_to_tree(
    root: ET.Element,
    measured: MeasuredPositionErrors,
    *,
    devices: tuple[str, ...] = IC_SIGMA_DEVICES,
    optimize_mode: PositionOptimizeMode = DEFAULT_SIGMA_OPTIMIZE_MODE,
) -> PositionOffsetTuneResult:
    """Set ``zero_offset_at_iso_mm`` from session position errors."""
    updates, warnings = collect_offset_updates(
        root,
        measured,
        devices=devices,
        optimize_mode=optimize_mode,
    )
    for update in updates:
        update.element.text = format_offset_mm(update.new_offset)
    return PositionOffsetTuneResult(offsets_updated=len(updates), warnings=warnings)


def tune_position_offsets_from_sessions(
    root: ET.Element,
    session_ids: list[str],
    base_dir: str,
    *,
    data_source: PositionDataSource = DEFAULT_POSITION_DATA_SOURCE,
    optimize_mode: PositionOptimizeMode = DEFAULT_SIGMA_OPTIMIZE_MODE,
) -> PositionOffsetTuneResult:
    """Load position errors from all sessions and apply them to *root*."""
    measured, load_warnings = load_measured_position_errors_for_sessions(
        session_ids,
        base_dir,
        data_source=data_source,
    )
    if measured is None:
        return PositionOffsetTuneResult(warnings=load_warnings)
    result = apply_position_offsets_to_tree(
        root,
        measured,
        optimize_mode=optimize_mode,
    )
    if load_warnings:
        result.warnings = load_warnings + result.warnings
    return result
