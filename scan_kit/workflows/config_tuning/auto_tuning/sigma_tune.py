"""Apply measured session sigmas to ``devices.xml`` beam_sigma_conversions."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from scan_kit.common.devices_xml import IC_SIGMA_DEVICES
from scan_kit.common.session_sigma import (
    MeasuredSigmaSpots,
    load_measured_sigma_spots,
    load_measured_sigma_spots_for_sessions,
)

SigmaOptimizeMode = Literal["median", "weighted_average", "min_max_midpoint"]

DEFAULT_SIGMA_OPTIMIZE_MODE: SigmaOptimizeMode = "median"


def format_sigma_k0(value: float) -> str:
    """Format K0 the way room ``devices.xml`` files typically store it."""
    v = float(value)
    if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 1e-4):
        return f"{v:.6E}"
    rounded = round(v, 3)
    if abs(rounded - v) < 1e-9:
        return f"{rounded:.3f}".rstrip("0").rstrip(".")
    return f"{v:.6g}"


def normalize_sigma_optimize_mode(value: str | None) -> SigmaOptimizeMode:
    if value in ("weighted_average", "min_max_midpoint", "median"):
        return value
    return DEFAULT_SIGMA_OPTIMIZE_MODE


def band_sigma_variance(sigmas: np.ndarray) -> float:
    """Sample variance (mm²) of observed spot sigmas in one energy band."""
    if sigmas.size < 2:
        return 0.0
    return float(np.var(sigmas, ddof=1))


def band_furthest_extreme_pct_deviation(
    sigmas: np.ndarray,
    new_k0: float,
) -> tuple[float, float, str]:
    """Percent deviation of the min/max extreme furthest from *new_k0*.

    Returns ``(abs_pct, observed_mm, kind)`` where *kind* is ``"min"`` or ``"max"``.
    """
    if sigmas.size == 0 or not np.isfinite(new_k0) or abs(new_k0) < 1e-12:
        return float("nan"), float("nan"), ""
    min_sigma = float(np.min(sigmas))
    max_sigma = float(np.max(sigmas))
    if abs(min_sigma - new_k0) >= abs(max_sigma - new_k0):
        observed = min_sigma
        kind = "min"
    else:
        observed = max_sigma
        kind = "max"
    pct = abs(observed - new_k0) / abs(new_k0) * 100.0
    return pct, observed, kind


def compute_band_sigma(
    sigmas: np.ndarray,
    weights: np.ndarray | None,
    mode: SigmaOptimizeMode,
) -> float:
    """Reduce spot sigmas in one energy band to a single calibration value."""
    if mode == "min_max_midpoint":
        return float((np.min(sigmas) + np.max(sigmas)) / 2.0)
    if mode == "weighted_average":
        if weights is not None:
            valid = np.isfinite(weights) & (weights > 0) & np.isfinite(sigmas)
            if np.any(valid):
                return float(np.average(sigmas[valid], weights=weights[valid]))
        return float(np.mean(sigmas))
    return float(np.median(sigmas))


@dataclass(frozen=True)
class SigmaTunePreviewRow:
    """One tunable energy band and its proposed K0 change."""

    device: str
    min_energy: float
    max_energy: float
    old_k0: float
    new_k0: float
    n_spots: int
    sigma_variance: float
    extreme_pct_deviation: float
    extreme_observed_mm: float
    extreme_kind: str

    @property
    def energy_center_mev(self) -> float:
        return (self.min_energy + self.max_energy) / 2.0

    @property
    def delta_k0(self) -> float:
        return self.new_k0 - self.old_k0


@dataclass
class SigmaTuneResult:
    bands_updated: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.bands_updated > 0


@dataclass
class _BandUpdate:
    element: ET.Element
    device: str
    min_energy: float
    max_energy: float
    old_k0: float
    new_k0: float
    n_spots: int
    sigma_variance: float
    extreme_pct_deviation: float
    extreme_observed_mm: float
    extreme_kind: str


_DEVICE_ORDER = {name: index for index, name in enumerate(IC_SIGMA_DEVICES)}


def collect_sigma_band_updates(
    root: ET.Element,
    measured: MeasuredSigmaSpots,
    *,
    devices: tuple[str, ...] = IC_SIGMA_DEVICES,
    optimize_mode: SigmaOptimizeMode = DEFAULT_SIGMA_OPTIMIZE_MODE,
) -> tuple[list[_BandUpdate], list[str]]:
    """Collect per-band K0 updates without mutating *root*."""
    updates: list[_BandUpdate] = []
    warnings: list[str] = []
    device_set = set(devices)

    for chamber in root.iter("ion_chamber"):
        device_el = chamber.find("device")
        if device_el is None:
            continue
        name = device_el.get("name")
        if not name or name not in device_set:
            continue
        spot_data = measured.by_device.get(name)
        if spot_data is None:
            warnings.append(f"No session sigma data for {name}.")
            continue
        energies, sigmas = spot_data

        for el in chamber.findall("beam_sigma_conversions"):
            if el.get("in_units", "").upper() != "MEV":
                continue
            if el.get("out_units", "").lower() != "mm":
                continue
            try:
                min_e = float(el.get("min_energy", "nan"))
                max_e = float(el.get("max_energy", "nan"))
                old_k0 = float(el.get("K0", "0"))
                k1 = float(el.get("K1", "0"))
                k2 = float(el.get("K2", "0"))
                k3 = float(el.get("K3", "0"))
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(min_e) and np.isfinite(max_e)):
                continue
            if abs(k1) > 1e-12 or abs(k2) > 1e-12 or abs(k3) > 1e-12:
                continue

            mask = (energies >= min_e) & (energies <= max_e)
            if not np.any(mask):
                continue
            band_sigmas = sigmas[mask]
            band_weights = measured.weights[mask] if measured.weights is not None else None
            new_k0 = compute_band_sigma(band_sigmas, band_weights, optimize_mode)
            if not np.isfinite(new_k0):
                continue
            extreme_pct, extreme_mm, extreme_kind = band_furthest_extreme_pct_deviation(
                band_sigmas,
                new_k0,
            )
            updates.append(
                _BandUpdate(
                    element=el,
                    device=name,
                    min_energy=min_e,
                    max_energy=max_e,
                    old_k0=old_k0,
                    new_k0=new_k0,
                    n_spots=int(np.count_nonzero(mask)),
                    sigma_variance=band_sigma_variance(band_sigmas),
                    extreme_pct_deviation=extreme_pct,
                    extreme_observed_mm=extreme_mm,
                    extreme_kind=extreme_kind,
                )
            )

    if not updates and not warnings:
        warnings.append("No beam_sigma_conversions bands matched session energies.")
    return updates, warnings


def preview_rows_from_updates(updates: list[_BandUpdate]) -> list[SigmaTunePreviewRow]:
    rows = [
        SigmaTunePreviewRow(
            device=u.device,
            min_energy=u.min_energy,
            max_energy=u.max_energy,
            old_k0=u.old_k0,
            new_k0=u.new_k0,
            n_spots=u.n_spots,
            sigma_variance=u.sigma_variance,
            extreme_pct_deviation=u.extreme_pct_deviation,
            extreme_observed_mm=u.extreme_observed_mm,
            extreme_kind=u.extreme_kind,
        )
        for u in updates
    ]
    rows.sort(
        key=lambda row: (
            _DEVICE_ORDER.get(row.device, 99),
            -row.energy_center_mev,
        )
    )
    return rows


def compute_sigma_tune_preview(
    root: ET.Element,
    session_ids: list[str],
    base_dir: str,
    *,
    optimize_mode: SigmaOptimizeMode = DEFAULT_SIGMA_OPTIMIZE_MODE,
) -> tuple[list[SigmaTunePreviewRow], list[str]]:
    """Return proposed K0 values for every matching band in *root*."""
    measured, load_warnings = load_measured_sigma_spots_for_sessions(session_ids, base_dir)
    if measured is None:
        return [], load_warnings
    updates, warnings = collect_sigma_band_updates(
        root,
        measured,
        optimize_mode=optimize_mode,
    )
    return preview_rows_from_updates(updates), load_warnings + warnings


def apply_measured_sigmas_to_tree(
    root: ET.Element,
    measured: MeasuredSigmaSpots,
    *,
    devices: tuple[str, ...] = IC_SIGMA_DEVICES,
    optimize_mode: SigmaOptimizeMode = DEFAULT_SIGMA_OPTIMIZE_MODE,
) -> SigmaTuneResult:
    """Set ``K0`` on constant (K1–K3 ≈ 0) bands from spot data in each energy band."""
    updates, warnings = collect_sigma_band_updates(
        root,
        measured,
        devices=devices,
        optimize_mode=optimize_mode,
    )
    for update in updates:
        update.element.set("K0", format_sigma_k0(update.new_k0))
    return SigmaTuneResult(bands_updated=len(updates), warnings=warnings)


def tune_sigmas_from_sessions(
    root: ET.Element,
    session_ids: list[str],
    base_dir: str,
    *,
    optimize_mode: SigmaOptimizeMode = DEFAULT_SIGMA_OPTIMIZE_MODE,
) -> SigmaTuneResult:
    """Load spot sigmas from all sessions and apply them to *root*."""
    measured, load_warnings = load_measured_sigma_spots_for_sessions(session_ids, base_dir)
    if measured is None:
        return SigmaTuneResult(warnings=load_warnings)
    result = apply_measured_sigmas_to_tree(root, measured, optimize_mode=optimize_mode)
    if load_warnings:
        result.warnings = load_warnings + result.warnings
    return result


def tune_sigmas_from_session(
    root: ET.Element,
    session_id: str,
    base_dir: str,
    *,
    optimize_mode: SigmaOptimizeMode = DEFAULT_SIGMA_OPTIMIZE_MODE,
) -> SigmaTuneResult:
    """Load one session's spot sigmas and apply them to *root*."""
    return tune_sigmas_from_sessions(
        root,
        [session_id],
        base_dir,
        optimize_mode=optimize_mode,
    )
