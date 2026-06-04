"""Parse session ``config/map2map/devices.xml`` device metadata."""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from .session_source import SessionSource, load_session_text, resolve_session_source

_log = logging.getLogger(__name__)

DEVICES_XML_REL_PATH = "config/map2map/devices.xml"

IC_SIGMA_DEVICES = ("IC_1_X", "IC_1_Y", "IC_2_X", "IC_2_Y")

IC_DEVICE_TO_SIG_KEY = {
    "IC_1_X": "ic1_sig_x",
    "IC_1_Y": "ic1_sig_y",
    "IC_2_X": "ic2_sig_x",
    "IC_2_Y": "ic2_sig_y",
}


@dataclass(frozen=True)
class BeamSigmaConversion:
    """Energy-band beam sigma calibration for one ion chamber."""

    min_energy: float
    max_energy: float
    k0: float
    k1: float
    k2: float
    k3: float

    def sigma_mm(self, energy_mev: float) -> float:
        e = float(energy_mev)
        return self.k0 + self.k1 * e + self.k2 * e * e + self.k3 * e * e * e


@dataclass
class DevicesConfig:
    """Parsed beam sigma expectations from devices.xml."""

    beam_sigmas: dict[str, list[BeamSigmaConversion]] = field(default_factory=dict)

    def expected_sigma_mm(self, device: str, energy_mev: float) -> float | None:
        conversions = self.beam_sigmas.get(device)
        if not conversions:
            return None
        e = float(energy_mev)
        for conv in conversions:
            if conv.min_energy <= e <= conv.max_energy:
                return conv.sigma_mm(e)
        return None

    def expected_sigmas_by_key(
        self,
        energies,
        *,
        keys: tuple[str, ...] | None = None,
    ) -> dict[str, dict[float, float]]:
        """Map sigma view keys (``ic1_sig_x``, …) to ``{energy: sigma_mm}``."""
        device_items = IC_DEVICE_TO_SIG_KEY.items()
        if keys is not None:
            key_set = set(keys)
            device_items = (
                (device, sig_key)
                for device, sig_key in IC_DEVICE_TO_SIG_KEY.items()
                if sig_key in key_set
            )

        result: dict[str, dict[float, float]] = {}
        for device, sig_key in device_items:
            per_energy: dict[float, float] = {}
            for energy in energies:
                sigma = self.expected_sigma_mm(device, energy)
                if sigma is not None and math.isfinite(sigma):
                    per_energy[float(energy)] = float(sigma)
            if per_energy:
                result[sig_key] = per_energy
        return result


def parse_devices_xml(text: str) -> DevicesConfig:
    """Parse devices.xml text into :class:`DevicesConfig`."""
    root = ET.fromstring(text)
    config = DevicesConfig()

    for chamber in root.iter("ion_chamber"):
        device_el = chamber.find("device")
        if device_el is None:
            continue
        name = device_el.get("name")
        if not name:
            continue

        conversions: list[BeamSigmaConversion] = []
        for el in chamber.findall("beam_sigma_conversions"):
            if el.get("in_units", "").upper() != "MEV":
                continue
            if el.get("out_units", "").lower() != "mm":
                continue
            try:
                conversions.append(
                    BeamSigmaConversion(
                        min_energy=float(el.get("min_energy", "nan")),
                        max_energy=float(el.get("max_energy", "nan")),
                        k0=float(el.get("K0", "0")),
                        k1=float(el.get("K1", "0")),
                        k2=float(el.get("K2", "0")),
                        k3=float(el.get("K3", "0")),
                    )
                )
            except (TypeError, ValueError):
                continue

        if conversions:
            config.beam_sigmas[name] = conversions

    return config


def load_devices_config(source: SessionSource) -> DevicesConfig | None:
    """Load and parse ``devices.xml`` for *source*."""
    text = load_session_text(source, DEVICES_XML_REL_PATH)
    if text is None:
        return None
    try:
        return parse_devices_xml(text)
    except ET.ParseError as exc:
        _log.debug("Session %s: devices.xml parse error: %s", source.session_id, exc)
        return None


def load_session_devices_config(
    session_id: str,
    base_dir: str | Path,
) -> DevicesConfig | None:
    """Resolve *session_id* under *base_dir* and load its devices.xml."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None
    return load_devices_config(src)
