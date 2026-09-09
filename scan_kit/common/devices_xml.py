"""Parse session ``config/map2map`` device metadata and isocenter geometry."""

from __future__ import annotations

import functools
import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .session_source import SessionSource, load_session_text, resolve_session_source

_log = logging.getLogger(__name__)

DEVICES_XML_REL_PATH = "config/map2map/devices.xml"
SYSTEM_XML_REL_PATH = "config/map2map/scan_dose_system.xml"

IC_SIGMA_DEVICES = ("IC_1_X", "IC_1_Y", "IC_2_X", "IC_2_Y")

# engine.cpp MAX_V_SAD / MAX_X_SAD / MAX_Y_SAD: load_system rejects anything larger.
MAX_SAD_MM = 6000.0

# ion_chamber.hpp convert_strips emits this for raw strip values <= -1.
INVALID_POSITION_MM = -10000.0

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


def _child_float(element: ET.Element, tag: str, default: float) -> float:
    child = element.find(tag)
    if child is None or child.text is None:
        return default
    try:
        return float(child.text.strip())
    except (TypeError, ValueError):
        return default


def _child_bool(element: ET.Element, tag: str, default: bool = False) -> bool:
    child = element.find(tag)
    if child is None or child.text is None:
        return default
    return child.text.strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class SystemGeometry:
    """The ``geometry`` block of ``scan_dose_system.xml``.

    Only ``virtual_sad_mm`` has any runtime effect: ``engine::set_mag_factors``
    feeds it to every ion chamber as the numerator of the magnification factor.
    ``sad_x_mm`` / ``sad_y_mm`` / the field sizes are range-checked by
    ``engine::load_system`` and then never read anywhere in ``scan_dose``.
    """

    virtual_sad_mm: float
    sad_x_mm: float = float("nan")
    sad_y_mm: float = float("nan")
    field_size_x_mm: float = float("nan")
    field_size_y_mm: float = float("nan")


@dataclass(frozen=True)
class IcGeometry:
    """One ion chamber's strip geometry from ``devices.xml``.

    ``xml_sad_mm`` is kept for display only.  map2map parses it, uses it to seed
    the magnification factor, then discards it (see :class:`Map2MapGeometry`).
    """

    name: str
    strip_count: float
    strip_to_mm: float
    zero_offset_at_iso_mm: float
    reverse_strips: bool
    sdd_mm: float
    xml_sad_mm: float = float("nan")

    @property
    def center_strip(self) -> float:
        """Zero-based center strip: C++ ``m_count / 2.0 - 0.5``."""
        return self.strip_count / 2.0 - 0.5

    def mag_factor(self, virtual_sad_mm: float) -> float:
        """Iso mm per IC mm. ``ion_chamber::set_mag_factor`` falls back to 1 at SDD 0."""
        if abs(self.sdd_mm) <= 1e-7:
            return 1.0
        return virtual_sad_mm / self.sdd_mm

    def iso_mm_per_strip(self, virtual_sad_mm: float) -> float:
        """Signed iso mm per strip; the sign carries ``reverse_strips``.

        Reversing strips as a sign flip is algebraically identical to the C++
        ``m_count - 1 - strips`` applied before centering.
        """
        sign = -1.0 if self.reverse_strips else 1.0
        return sign * self.strip_to_mm * self.mag_factor(virtual_sad_mm)

    def strip_to_iso(self, strips, virtual_sad_mm: float) -> np.ndarray:
        """Strip index to mm at isocenter, mirroring ``convert_strips`` for positions."""
        s = np.asarray(strips, dtype=float)
        iso = (
            self.iso_mm_per_strip(virtual_sad_mm) * (s - self.center_strip)
            + self.zero_offset_at_iso_mm
        )
        return np.where(s <= -1.0, INVALID_POSITION_MM, iso)

    def sigma_to_iso(self, sigma_strips, virtual_sad_mm: float) -> np.ndarray:
        """Spot size to mm at isocenter: gain and magnification only, no offset."""
        scale = abs(self.strip_to_mm * self.mag_factor(virtual_sad_mm))
        return scale * np.asarray(sigma_strips, dtype=float)


@dataclass(frozen=True)
class MagnetGeometry:
    """Scan-magnet geometry from ``devices.xml``.

    ``axis_to_iso_mm`` is the magnet's own SAD (``scan_magnet.cpp`` ``m_sad``,
    exposed as ``get_sad``).  ``set_mag_factors`` is a no-op for scan magnets, so
    the system ``virtual_sad`` never touches this value.
    """

    name: str
    axis: str
    axis_to_iso_mm: float


@dataclass
class DevicesConfig:
    """Parsed beam sigma expectations and device geometry from devices.xml."""

    beam_sigmas: dict[str, list[BeamSigmaConversion]] = field(default_factory=dict)
    chambers: dict[str, IcGeometry] = field(default_factory=dict)
    magnets: dict[str, MagnetGeometry] = field(default_factory=dict)

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

        geometry = _parse_ic_geometry(name, chamber)
        if geometry is not None:
            config.chambers[name] = geometry

    for magnet in root.iter("scan_magnet"):
        device_el = magnet.find("device")
        if device_el is None:
            continue
        name = device_el.get("name")
        if not name:
            continue
        axis_to_iso = _child_float(magnet, "magnet_axis_to_iso_distance_mm", float("nan"))
        if not math.isfinite(axis_to_iso):
            continue
        config.magnets[name] = MagnetGeometry(
            name=name,
            axis=(magnet.findtext("major_axis") or "").strip().lower(),
            axis_to_iso_mm=axis_to_iso,
        )

    return config


def _parse_ic_geometry(name: str, chamber: ET.Element) -> IcGeometry | None:
    """Build :class:`IcGeometry`, skipping chambers map2map itself would reject.

    ``strip_count``, ``strip_to_mm`` and ``source_to_device_distance_mm`` are read
    with no default in ``ion_chamber::set_data``, so a chamber missing any of them
    fails the real config load; there is no safe value to invent.
    """
    strip_count = _child_float(chamber, "strip_count", float("nan"))
    strip_to_mm = _child_float(chamber, "strip_to_mm", float("nan"))
    sdd_mm = _child_float(chamber, "source_to_device_distance_mm", float("nan"))
    if not (math.isfinite(strip_count) and math.isfinite(strip_to_mm) and math.isfinite(sdd_mm)):
        return None
    if strip_to_mm < 0.1:
        # ion_chamber.cpp clamps implausible strip pitch to 1 mm.
        strip_to_mm = 1.0
    return IcGeometry(
        name=name,
        strip_count=strip_count,
        strip_to_mm=strip_to_mm,
        zero_offset_at_iso_mm=_child_float(chamber, "zero_offset_at_iso_mm", 0.0),
        reverse_strips=_child_bool(chamber, "reverse_strips"),
        sdd_mm=sdd_mm,
        xml_sad_mm=_child_float(chamber, "source_to_axis_distance_mm", float("nan")),
    )


def parse_system_geometry(text: str) -> SystemGeometry | None:
    """Parse ``scan_dose_system.xml`` text, applying ``load_system``'s range checks."""
    geom = ET.fromstring(text).find("geometry")
    if geom is None:
        return None
    virtual_sad = _child_float(geom, "source_to_isocenter_distance", float("nan"))
    if not (math.isfinite(virtual_sad) and 0.0 < virtual_sad <= MAX_SAD_MM):
        return None
    return SystemGeometry(
        virtual_sad_mm=virtual_sad,
        sad_x_mm=_child_float(geom, "source_to_x_axis_distance", float("nan")),
        sad_y_mm=_child_float(geom, "source_to_y_axis_distance", float("nan")),
        field_size_x_mm=_child_float(geom, "field_size_x", float("nan")),
        field_size_y_mm=_child_float(geom, "field_size_y", float("nan")),
    )


@dataclass(frozen=True)
class Map2MapGeometry:
    """Strip/sigma conversions exactly as map2map applies them at runtime.

    ``engine::load_configuration`` ends with ``set_mag_factors(m_geom.virtual_sad)``,
    which overwrites the per-chamber ``source_to_axis_distance_mm`` seed for *every*
    ion chamber.  Magnification is therefore
    ``source_to_isocenter_distance / source_to_device_distance_mm``, and nothing here
    ever reads :attr:`IcGeometry.xml_sad_mm`.
    """

    system: SystemGeometry
    chambers: dict[str, IcGeometry] = field(default_factory=dict)
    magnets: dict[str, MagnetGeometry] = field(default_factory=dict)

    def mag_factor(self, device: str) -> float | None:
        chamber = self.chambers.get(device)
        if chamber is None:
            return None
        return chamber.mag_factor(self.system.virtual_sad_mm)

    def iso_mm_per_strip(self, device: str) -> float | None:
        chamber = self.chambers.get(device)
        if chamber is None:
            return None
        return chamber.iso_mm_per_strip(self.system.virtual_sad_mm)

    def strip_to_iso(self, device: str, strips) -> np.ndarray | None:
        chamber = self.chambers.get(device)
        if chamber is None:
            return None
        return chamber.strip_to_iso(strips, self.system.virtual_sad_mm)

    def sigma_to_iso(self, device: str, sigma_strips) -> np.ndarray | None:
        chamber = self.chambers.get(device)
        if chamber is None:
            return None
        return chamber.sigma_to_iso(sigma_strips, self.system.virtual_sad_mm)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def load_map2map_geometry(config_dir: Path | str) -> Map2MapGeometry | None:
    """Load geometry from a ``config/map2map`` directory.

    Returns ``None`` when either file is missing or unparsable — the same outcome
    map2map has, since ``load_configuration`` needs both.
    """
    config_path = Path(config_dir)
    devices_text = _read_text(config_path / "devices.xml")
    system_text = _read_text(config_path / "scan_dose_system.xml")
    if devices_text is None or system_text is None:
        return None
    try:
        devices = parse_devices_xml(devices_text)
        system = parse_system_geometry(system_text)
    except ET.ParseError as exc:
        _log.debug("map2map config parse error under %s: %s", config_path, exc)
        return None
    if system is None or not devices.chambers:
        return None
    return Map2MapGeometry(
        system=system,
        chambers=devices.chambers,
        magnets=devices.magnets,
    )


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


@functools.lru_cache(maxsize=64)
def load_session_devices_config(
    session_id: str,
    base_dir: str,
) -> DevicesConfig | None:
    """Resolve *session_id* under *base_dir* and load its devices.xml."""
    src = resolve_session_source(session_id, str(base_dir))
    if src is None:
        return None
    return load_devices_config(src)
