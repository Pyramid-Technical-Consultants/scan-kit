"""Apply measured session sigmas to ``devices.xml`` beam_sigma_conversions."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from scan_kit.common.devices_xml import IC_SIGMA_DEVICES
from scan_kit.common.session_sigma import load_measured_sigma_spots


def format_sigma_k0(value: float) -> str:
    """Format K0 the way room ``devices.xml`` files typically store it."""
    v = float(value)
    if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 1e-4):
        return f"{v:.6E}"
    rounded = round(v, 3)
    if abs(rounded - v) < 1e-9:
        return f"{rounded:.3f}".rstrip("0").rstrip(".")
    return f"{v:.6g}"


@dataclass
class SigmaTuneResult:
    bands_updated: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.bands_updated > 0


def apply_measured_sigmas_to_tree(
    root: ET.Element,
    measured_spots: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    devices: tuple[str, ...] = IC_SIGMA_DEVICES,
) -> SigmaTuneResult:
    """Set ``K0`` on constant (K1–K3 ≈ 0) bands from spot data in each energy band."""
    result = SigmaTuneResult()
    device_set = set(devices)

    for chamber in root.iter("ion_chamber"):
        device_el = chamber.find("device")
        if device_el is None:
            continue
        name = device_el.get("name")
        if not name or name not in device_set:
            continue
        spot_data = measured_spots.get(name)
        if spot_data is None:
            result.warnings.append(f"No session sigma data for {name}.")
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
            new_k0 = float(np.median(sigmas[mask]))
            if not np.isfinite(new_k0):
                continue
            el.set("K0", format_sigma_k0(new_k0))
            result.bands_updated += 1

    if result.bands_updated == 0 and not result.warnings:
        result.warnings.append("No beam_sigma_conversions bands matched session energies.")
    return result


def tune_sigmas_from_session(
    root: ET.Element,
    session_id: str,
    base_dir: str,
) -> SigmaTuneResult:
    """Load session spot sigmas and apply them to *root*."""
    measured = load_measured_sigma_spots(session_id, base_dir)
    if measured is None:
        return SigmaTuneResult(
            warnings=[f"Could not load sigma columns for session {session_id!r}."],
        )
    return apply_measured_sigmas_to_tree(root, measured)
