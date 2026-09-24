"""Load IC Gaussians and map them through a pluggable depth axis."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

from ..common import (
    C_CHARGE_REQ,
    C_ENERGY,
    C_X_POSITION,
    C_Y_POSITION,
    load_session_raw,
    process_position_data,
    resolve_concept_column,
    resolve_session_source,
    try_load_position_data,
)
from ..common.devices_xml import (
    DEVICES_XML_REL_PATH,
    INVALID_POSITION_MM,
    SYSTEM_XML_REL_PATH,
    DevicesConfig,
    Map2MapGeometry,
    load_session_devices_config,
    parse_devices_xml,
    parse_system_geometry,
)
from ..common.ic_trajectory import IC1_Z_MM, IC2_Z_MM, IC_SEP_MM, ic_alignment_offsets
from ..common.scan_magnet_model import combined_isocenter_z
from ..common.session_sigma import IC_SIGMA_LABELS, resolve_spot_sigma_column
from ..common.session_source import load_session_text
from ..common.timeslice_sigma import (
    TIMESLICE_SIGMA_COLS,
    frame_timeslice_sigma_arrays,
    resolve_timeslice_sigma_source,
)
from ..common.timeslice_table import load_energy_tagged_table
from ..common.trajectory_fits import fit_iso_plane, fit_magnet_pivot
from ..data.reference_frame import timeslice_position_table_hooks
from ..data.types import REFERENCE_CHAMBER, REFERENCE_ISO
from .dose_volume_physics import csda_range_mm, depth_sigma_mm, medium_for
from .dose_volume_catalog import (
    GRAIN_SPOT,
    GRAIN_TIMESLICE,
    MEDIUM_COPPER,
    MEDIUM_WATER,
    PLAN_SIGMA_INTERLOCK,
    PLAN_SIGMA_REFERENCE,
    SIGMA_PLANE_CHAMBER,
    SIGMA_PLANE_ISO,
    XY_IC1,
    XY_IC2,
    XY_ISO_RAY,
    XY_PLAN,
    SplatConfig,
)

_FALLBACK_SIGMA_MM = 4.0
# Logged spot σ is in strips; IC strip pitch in mm, as the other sigma views assume.
_SIGMA_STRIP_MM = 2.0
_ABS_INVALID_MM = abs(INVALID_POSITION_MM) * 0.9
# ponytail: typical PBS IC when devices.xml has no K_MU. Read the session value first.
_FALLBACK_KMU_C_PER_MU = 2.0e-8
# Ideal air IC, no recombination (ICRU 90 W, 20 °C dry air).
_W_AIR_EV = 33.97
_E_CHARGE_C = 1.602176634e-19
_RHO_AIR_G_CM3 = 1.205e-3
# ponytail: power-law fit to PSTAR air at 70 and 230 MeV; ICRU table if counts look off.
_PSTAR_S70 = 9.20
_PSTAR_E_REF = 70.0
_PSTAR_S_EXP = -0.643


class DepthAxis(Protocol):
    """Maps energy (MeV) to scene-mm depth (range; +Z is up, beam from the sky)."""

    axis_label: str
    smear_label: str
    # +1 if +Z is deeper / higher energy; −1 if depth grows downward (beam from sky).
    depth_sign: int

    def z_scene_mm(self, energy_mev: np.ndarray) -> np.ndarray: ...

    def sigma_z_scene_mm(
        self, energy_mev: np.ndarray, smear_axis_units: float,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class LinearEnergyAxis:
    """Linear MeV→mm (tests / protocol). Live view uses :class:`RangeAxis`."""

    mm_per_mev: float
    e_min: float = 0.0
    axis_label: str = "Energy (MeV)"
    smear_label: str = "MeV"
    depth_sign: int = 1

    def z_scene_mm(self, energy_mev: np.ndarray) -> np.ndarray:
        return (np.asarray(energy_mev, dtype=float) - self.e_min) * self.mm_per_mev

    def sigma_z_scene_mm(
        self, energy_mev: np.ndarray, smear_axis_units: float,
    ) -> np.ndarray:
        thickness = abs(float(smear_axis_units) * self.mm_per_mev)
        return np.full(np.shape(energy_mev), max(thickness, 1e-6), dtype=float)


@dataclass(frozen=True)
class RangeAxis:
    """CSDA proton range from Bethe stopping-power tables (mm, MeV).

    Scene +Z is up. Depth is ``−R`` so higher energy is deeper *and* sits at
    the bottom of the stack at gantry 0° (beam from the sky). The guide axis
    points the same way (``depth_sign = −1``). The smear argument is the
    beam's energy spread σE in %; σz adds range straggling in quadrature.
    """

    medium: str
    axis_label: str = "Depth in water (mm)"
    smear_label: str = "%"
    depth_sign: int = -1

    def z_scene_mm(self, energy_mev: np.ndarray) -> np.ndarray:
        e = np.maximum(np.asarray(energy_mev, dtype=float), 1.0)
        return -csda_range_mm(medium_for(self.medium), e)

    def sigma_z_scene_mm(
        self, energy_mev: np.ndarray, smear_axis_units: float,
    ) -> np.ndarray:
        e = np.maximum(np.asarray(energy_mev, dtype=float), 1.0)
        return np.maximum(depth_sigma_mm(medium_for(self.medium), e, float(smear_axis_units)), 1e-6)


def range_axis_for_medium(medium: str) -> RangeAxis:
    key = MEDIUM_COPPER if medium == MEDIUM_COPPER else MEDIUM_WATER
    return RangeAxis(medium=key, axis_label=f"Depth in {key} (mm)")


@dataclass(frozen=True)
class SplatCloud:
    x: np.ndarray
    y: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    energy: np.ndarray
    weight: np.ndarray
    k_mu: float | None = None
    dose: np.ndarray | None = None


@dataclass(frozen=True)
class SplatBatch:
    center: np.ndarray
    sigma: np.ndarray
    weight: np.ndarray
    energy_mev: np.ndarray
    dose_mu: np.ndarray | None = None
    k_mu: float | None = None


@dataclass(frozen=True)
class PositionSigmaFrame:
    energy: np.ndarray
    ic1_x: np.ndarray
    ic1_y: np.ndarray
    ic2_x: np.ndarray
    ic2_y: np.ndarray
    ic1_sx: np.ndarray
    ic1_sy: np.ndarray
    ic2_sx: np.ndarray
    ic2_sy: np.ndarray
    weight: np.ndarray
    plan_x: np.ndarray | None = None
    plan_y: np.ndarray | None = None
    dose_ic1: np.ndarray | None = None
    dose_ic2: np.ndarray | None = None


@dataclass(frozen=True)
class SessionSplatSource:
    session_id: str
    iso: PositionSigmaFrame | None
    chamber: PositionSigmaFrame | None
    plan: SplatCloud | None
    n_raw: int
    geom: Map2MapGeometry | None = None
    k_mu: float | None = None


def iso_xy_from_ic_ray(
    ic2_xy: np.ndarray,
    ic1_xy: np.ndarray,
    z_iso: float,
    *,
    z_ic2: float = IC2_Z_MM,
    z_ic1: float = IC1_Z_MM,
) -> np.ndarray:
    """Lateral position on the IC2→IC1 ray at beamline *z_iso*."""
    p2 = np.asarray(ic2_xy, dtype=float)
    p1 = np.asarray(ic1_xy, dtype=float)
    denom = z_ic1 - z_ic2
    t = (float(z_iso) - z_ic2) / denom if denom != 0.0 else 1.0
    return p2 + t * (p1 - p2)


def concat_clouds(clouds: Sequence[SplatCloud]) -> SplatCloud | None:
    parts = [c for c in clouds if c.x.size]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return SplatCloud(
        x=np.concatenate([c.x for c in parts]),
        y=np.concatenate([c.y for c in parts]),
        sx=np.concatenate([c.sx for c in parts]),
        sy=np.concatenate([c.sy for c in parts]),
        energy=np.concatenate([c.energy for c in parts]),
        weight=np.concatenate([c.weight for c in parts]),
        k_mu=parts[0].k_mu if len({c.k_mu for c in parts}) == 1 else None,
        dose=(
            np.concatenate([c.dose for c in parts])
            if all(c.dose is not None for c in parts) else None
        ),
    )


def apply_splat_cap(cloud: SplatCloud, cap: int) -> SplatCloud:
    n = int(cloud.x.size)
    if cap <= 0 or n <= cap:
        return cloud
    stride = int(np.ceil(n / cap))
    sl = slice(None, None, stride)
    return SplatCloud(
        x=cloud.x[sl],
        y=cloud.y[sl],
        sx=cloud.sx[sl],
        sy=cloud.sy[sl],
        energy=cloud.energy[sl],
        weight=cloud.weight[sl],
        k_mu=cloud.k_mu,
        dose=None if cloud.dose is None else cloud.dose[sl],
    )


def _sanitize_xy(arr) -> np.ndarray:
    out = np.asarray(arr, dtype=float).reshape(-1)
    out = out.copy()
    out[~np.isfinite(out)] = np.nan
    out[np.abs(out) >= _ABS_INVALID_MM] = np.nan
    return out


def _sanitize_sigma(arr, *, scale: float = 1.0) -> np.ndarray:
    out = np.asarray(arr, dtype=float).reshape(-1) * scale
    out = out.copy()
    out[~np.isfinite(out)] = np.nan
    out[out <= 0.0] = np.nan
    return out


def _as_weight(arr, n: int) -> np.ndarray:
    if arr is None:
        return np.ones(n, dtype=float)
    out = np.asarray(arr, dtype=float).reshape(-1)
    if out.size != n:
        return np.ones(n, dtype=float)
    bad = ~np.isfinite(out) | (out <= 0.0)
    out = out.copy()
    out[bad] = 1.0
    return out


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def air_mass_stopping_mev_cm2_g(energy_mev) -> np.ndarray:
    """Proton mass stopping power in dry air (MeV cm²/g)."""
    e = np.maximum(np.asarray(energy_mev, dtype=float), 1.0)
    return _PSTAR_S70 * np.power(e / _PSTAR_E_REF, _PSTAR_S_EXP)


def protons_from_charge_c(charge_c, energy_mev, gap_mm: float) -> np.ndarray:
    """Ideal parallel-plate air IC, collection efficiency 1: ``N = Q W / (e S L)``."""
    gap_cm = max(float(gap_mm), 1e-6) / 10.0
    de_mev = air_mass_stopping_mev_cm2_g(energy_mev) * _RHO_AIR_G_CM3 * gap_cm
    w_mev = _W_AIR_EV * 1e-6
    q_per_proton = (de_mev / w_mev) * _E_CHARGE_C
    return np.asarray(charge_c, dtype=float) / np.maximum(q_per_proton, 1e-40)


def protons_from_mu(mu, energy_mev, gap_mm: float, k_mu_c_per_mu: float | None) -> np.ndarray:
    k_mu = float(k_mu_c_per_mu) if k_mu_c_per_mu else _FALLBACK_KMU_C_PER_MU
    return protons_from_charge_c(
        np.asarray(mu, dtype=float) * k_mu, energy_mev, gap_mm,
    )


def parse_kmu_c_per_mu(devices_xml: str, ic: str = "IC_1") -> float | None:
    """``K_MU`` in C/MU of *ic*'s dose plate (``_HCC``), else any *ic* device, else any.

    Strip planes carry their own ``K_MU``; logged MU comes from the HCC.
    """
    try:
        root = ET.fromstring(devices_xml)
    except ET.ParseError:
        return None
    hcc: list[float] = []
    preferred: list[float] = []
    others: list[float] = []
    for chamber in root.iter("ion_chamber"):
        device_el = chamber.find("device")
        name = (device_el.get("name") or "").strip() if device_el is not None else ""
        for conv in chamber.iter("gain_conversion"):
            if (conv.get("in_units") or "").upper() != "MU":
                continue
            raw = conv.get("K_MU") or conv.get("k_mu")
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val) or val == 0.0:
                continue
            upper = name.upper()
            if upper == f"{ic}_HCC":
                hcc.append(val)
            (preferred if upper.startswith(ic) else others).append(val)
            break
    if hcc:
        return float(hcc[0])
    if preferred:
        return float(preferred[0])
    if others:
        return float(others[0])
    return None


def session_kmu_c_per_mu(session_id: str, base_dir: str, ic: str = "IC_1") -> float | None:
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None
    text = load_session_text(src, DEVICES_XML_REL_PATH)
    if not text:
        return None
    return parse_kmu_c_per_mu(text, ic)


_SCAN_DOSE_COLS = (("ic1", "ic1_scan_total_dose"), ("ic2", "ic2_scan_total_dose"))
# G3 logs the scan-dose counter in nC, G2 in coulombs. A session under 1 mC
# (about 38 000 MU) must be coulombs; in nC it would be under 1 pC.
_COULOMB_SESSION_MAX = 1e-3


def timeslice_charge_nc(counter) -> np.ndarray:
    """Charge each timeslice added, from the layer's running scan-dose counter (nC)."""
    v = np.asarray(counter, dtype=float).reshape(-1)
    if v.size == 0:
        return v
    finite = np.isfinite(v)
    if not finite.any():
        return np.full(v.size, np.nan)
    # Hold the last reading across gaps so a missing sample does not drop or double charge.
    v = v[np.maximum.accumulate(np.where(finite, np.arange(v.size), 0))]
    v[: np.argmax(finite)] = v[np.argmax(finite)]
    return np.diff(v, prepend=v[0])


def fill_from_spot(values, spot, energy) -> np.ndarray:
    """Samples with no fit take their spot's median; a spot with none, the next sample in its layer.

    A spot's first slices ramp up before the IC fit locks on. They carry real charge,
    so dropping them would lose dose.
    """
    # ponytail: next-in-layer puts a spot with no fit at its neighbor, about one spot pitch off.
    s = pd.Series(np.asarray(values, dtype=float))
    layer = pd.Series(np.asarray(energy, dtype=float))
    out = s.fillna(s.groupby([layer, pd.Series(spot)]).transform("median"))
    by_layer = out.groupby(layer)
    return by_layer.bfill().fillna(by_layer.ffill()).to_numpy()


def nonnegative_layer_mu(mu, energy) -> np.ndarray:
    """Per-slice MU with noise below 0 clipped, each energy layer rescaled to keep its total."""
    raw = np.nan_to_num(np.asarray(mu, dtype=float), nan=0.0)
    out = np.clip(raw, 0.0, None)
    for e in np.unique(energy):
        m = energy == e
        kept = out[m].sum()
        if kept > 0.0:
            out[m] *= max(raw[m].sum(), 0.0) / kept
    return out


def cloud_to_batch(cloud: SplatCloud, axis: DepthAxis, smear_axis_units: float) -> SplatBatch:
    z = axis.z_scene_mm(cloud.energy)
    sz = axis.sigma_z_scene_mm(cloud.energy, smear_axis_units)
    mask = _finite_mask(cloud.x, cloud.y, cloud.sx, cloud.sy, cloud.energy, z, sz)
    center = np.column_stack([cloud.x[mask], cloud.y[mask], z[mask]]).astype(np.float32)
    sigma = np.column_stack([cloud.sx[mask], cloud.sy[mask], sz[mask]]).astype(np.float32)
    dose = None if cloud.dose is None else np.asarray(cloud.dose[mask], dtype=np.float32)
    return SplatBatch(
        center=center,
        sigma=sigma,
        weight=np.asarray(cloud.weight[mask], dtype=np.float32),
        energy_mev=np.asarray(cloud.energy[mask], dtype=np.float32),
        dose_mu=dose,
        k_mu=cloud.k_mu,
    )


def load_session_map2map_geometry(session_id: str, base_dir: str) -> Map2MapGeometry | None:
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None
    devices_text = load_session_text(src, DEVICES_XML_REL_PATH)
    system_text = load_session_text(src, SYSTEM_XML_REL_PATH)
    if devices_text is None or system_text is None:
        return None
    try:
        devices = parse_devices_xml(devices_text)
        system = parse_system_geometry(system_text)
    except Exception:
        return None
    if system is None or not devices.chambers:
        return None
    return Map2MapGeometry(
        system=system,
        chambers=devices.chambers,
        magnets=devices.magnets,
    )


def iso_z_from_map2map(geom: Map2MapGeometry) -> float:
    sad = geom.system.virtual_sad_mm
    for name in ("IC_2_X", "IC_2_Y"):
        chamber = geom.chambers.get(name)
        if chamber is not None and np.isfinite(chamber.sdd_mm) and chamber.sdd_mm > 0:
            return float(sad - chamber.sdd_mm)
    return float("nan")


def resolve_iso_z_mm(
    frame: PositionSigmaFrame,
    geom: Map2MapGeometry | None,
) -> float:
    if frame.plan_x is not None and frame.plan_y is not None:
        mx = fit_magnet_pivot(frame.ic2_x, frame.ic1_x)
        my = fit_magnet_pivot(frame.ic2_y, frame.ic1_y)
        iso_x = fit_iso_plane(frame.ic2_x, frame.ic1_x, frame.plan_x, mx.z_pivot) if mx.is_valid else None
        iso_y = fit_iso_plane(frame.ic2_y, frame.ic1_y, frame.plan_y, my.z_pivot) if my.is_valid else None
        z = combined_isocenter_z(iso_x, iso_y)
        if np.isfinite(z):
            return float(z)
    if geom is not None:
        z = iso_z_from_map2map(geom)
        if np.isfinite(z):
            return float(z)
    return IC1_Z_MM


def iso_sigma_scale(geom: Map2MapGeometry | None, t: float, axis: str = "X") -> float:
    """Chamber-to-isocenter σ scale on the IC2→IC1 ray at *t* (0 at IC2, 1 at IC1)."""
    # ponytail: linear interpolate chamber mag; magnet-pivot drift is the upgrade.
    if geom is None:
        return 1.0
    m1 = geom.mag_factor(f"IC_1_{axis}")
    m2 = geom.mag_factor(f"IC_2_{axis}")
    if m1 is None and m2 is None:
        return 1.0
    a = float(m2) if m2 is not None else 1.0
    b = float(m1) if m1 is not None else 1.0
    return (1.0 - t) * a + t * b


def _to_plane(cloud: SplatCloud, geom: Map2MapGeometry | None, t: float, plane: str) -> SplatCloud:
    """Chamber-mm σ of *cloud* drawn at *plane*; *t* places it on the IC2→IC1 ray."""
    if plane != SIGMA_PLANE_ISO:
        return cloud
    return replace(
        cloud,
        sx=cloud.sx * iso_sigma_scale(geom, t, "X"),
        sy=cloud.sy * iso_sigma_scale(geom, t, "Y"),
    )


def layer_sigmas(cloud: SplatCloud, energy: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Median σ of each energy layer in *cloud*, interpolated to *energy*."""
    # ponytail: flat beyond the cloud's energies; a σ(E) beam model is the upgrade.
    ok = np.isfinite(cloud.sx) & np.isfinite(cloud.sy) & np.isfinite(cloud.energy)
    if not ok.any():
        return None
    layers, inv = np.unique(np.round(cloud.energy[ok], 2), return_inverse=True)
    sx = np.array([np.median(cloud.sx[ok][inv == i]) for i in range(layers.size)])
    sy = np.array([np.median(cloud.sy[ok][inv == i]) for i in range(layers.size)])
    e = np.asarray(energy, dtype=float)
    return np.interp(e, layers, sx), np.interp(e, layers, sy)


def expected_plan_sigmas(
    devices: DevicesConfig | None,
    energy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(energy, dtype=float)
    sx = np.full(e.shape, _FALLBACK_SIGMA_MM, dtype=float)
    sy = np.full(e.shape, _FALLBACK_SIGMA_MM, dtype=float)
    if devices is None:
        return sx, sy
    for i, mev in enumerate(e):
        vx = devices.expected_sigma_mm("IC_1_X", float(mev))
        vy = devices.expected_sigma_mm("IC_1_Y", float(mev))
        if vx is not None:
            sx[i] = vx
        if vy is not None:
            sy[i] = vy
    return sx, sy


def _plan_from_input_map(session_id: str, base_dir: str) -> SplatCloud | None:
    input_map, _spot = load_session_raw(session_id, base_dir=base_dir)
    if input_map is None:
        return None
    e_col = resolve_concept_column(input_map.columns, C_ENERGY)
    x_col = resolve_concept_column(input_map.columns, C_X_POSITION)
    y_col = resolve_concept_column(input_map.columns, C_Y_POSITION)
    if e_col is None or x_col is None or y_col is None:
        return None
    energy = np.asarray(input_map[e_col], dtype=float)
    x = _sanitize_xy(input_map[x_col])
    y = _sanitize_xy(input_map[y_col])
    w_col = resolve_concept_column(input_map.columns, C_CHARGE_REQ)
    weight = _as_weight(input_map[w_col] if w_col is not None else None, energy.size)
    devices = load_session_devices_config(session_id, base_dir)
    sx, sy = expected_plan_sigmas(devices, energy)
    return SplatCloud(
        x=x, y=y, sx=sx, sy=sy, energy=energy, weight=weight,
        k_mu=session_kmu_c_per_mu(session_id, base_dir),
    )


_DOSE_CANDIDATES = (
    ("ic1", ("ic1_total_dose_spot", "ic1_total_dose")),
    ("ic2", ("ic2_total_dose_spot", "ic2_total_dose")),
)


def measured_dose_columns(columns) -> dict[str, str]:
    """Processed MU columns only. ``*_raw`` is nanocoulombs and would fake a huge error."""
    from ..common.schema import resolve_column_name

    found: dict[str, str] = {}
    for ic, names in _DOSE_CANDIDATES:
        for name in names:
            col = resolve_column_name(columns, name)
            if col is not None and "_raw" not in col.lower():
                found[ic] = col
                break
    return found


def _sigma_columns(spot_columns) -> dict[str, str]:
    found: dict[str, str] = {}
    for label, ic, axis in IC_SIGMA_LABELS:
        col = resolve_spot_sigma_column(spot_columns, ic, axis)
        if col is not None:
            found[label] = col
    return found


def _as_dose(data: dict, col: str | None, n: int) -> np.ndarray | None:
    if not col or col not in data:
        return None
    out = np.asarray(data[col], dtype=float).reshape(-1)
    if out.size != n:
        return None
    return out


def _frame_dose(frame: PositionSigmaFrame, ic: str) -> np.ndarray | None:
    if ic == "ic2":
        return frame.dose_ic2
    if ic == "ic1":
        return frame.dose_ic1
    if frame.dose_ic1 is not None and np.isfinite(frame.dose_ic1).any():
        return frame.dose_ic1
    return frame.dose_ic2


def _frame_from_position_data(
    data: dict,
    sigma_cols: dict[str, str],
    *,
    sigma_scale: float,
    dose_cols: dict[str, str] | None = None,
) -> PositionSigmaFrame:
    energy = np.asarray(data["energy"], dtype=float)
    n = energy.size
    sig = {label: _sanitize_sigma(data[col], scale=sigma_scale)
           if col in data else np.full(n, np.nan)
           for label, col in sigma_cols.items()}
    plan_x = _sanitize_xy(data[C_X_POSITION]) if C_X_POSITION in data else None
    plan_y = _sanitize_xy(data[C_Y_POSITION]) if C_Y_POSITION in data else None
    return PositionSigmaFrame(
        energy=energy,
        ic1_x=_sanitize_xy(data["ic1_x"]),
        ic1_y=_sanitize_xy(data["ic1_y"]),
        ic2_x=_sanitize_xy(data["ic2_x"]),
        ic2_y=_sanitize_xy(data["ic2_y"]),
        ic1_sx=sig.get("ic1_sig_x", np.full(n, np.nan)),
        ic1_sy=sig.get("ic1_sig_y", np.full(n, np.nan)),
        ic2_sx=sig.get("ic2_sig_x", np.full(n, np.nan)),
        ic2_sy=sig.get("ic2_sig_y", np.full(n, np.nan)),
        weight=_as_weight(data.get(C_CHARGE_REQ), n),
        plan_x=plan_x,
        plan_y=plan_y,
        dose_ic1=_as_dose(data, (dose_cols or {}).get("ic1"), n),
        dose_ic2=_as_dose(data, (dose_cols or {}).get("ic2"), n),
    )


def _load_spot_frame(
    session_id: str,
    base_dir: str,
    *,
    raw: bool,
    sigma_cols: dict[str, str],
    dose_cols: dict[str, str] | None = None,
) -> PositionSigmaFrame | None:
    extra_spot = list(sigma_cols.values()) + list((dose_cols or {}).values())

    def _loader(sid, position_key, bdir):
        return process_position_data(
            sid,
            position_key,
            extra_spot_columns=extra_spot,
            extra_input_columns=[C_X_POSITION, C_Y_POSITION, C_CHARGE_REQ],
            base_dir=bdir,
        )

    data = try_load_position_data(session_id, base_dir, _loader, raw=raw)
    if data is None:
        return None
    return _frame_from_position_data(
        data, sigma_cols, sigma_scale=_SIGMA_STRIP_MM, dose_cols=dose_cols,
    )


def _load_spot_source(session_id: str, base_dir: str) -> SessionSplatSource | None:
    _input_map, spot_data = load_session_raw(session_id, base_dir=base_dir)
    if spot_data is None:
        return None
    sigma_cols = _sigma_columns(spot_data.columns)
    dose_cols = measured_dose_columns(spot_data.columns)
    iso = _load_spot_frame(
        session_id, base_dir, raw=False, sigma_cols=sigma_cols, dose_cols=dose_cols,
    )
    chamber = _load_spot_frame(
        session_id, base_dir, raw=True, sigma_cols=sigma_cols, dose_cols=dose_cols,
    )
    if iso is None and chamber is None:
        return None
    plan = _plan_from_input_map(session_id, base_dir)
    n_raw = int((iso or chamber).energy.size)
    geom = load_session_map2map_geometry(session_id, base_dir)
    k_mu = plan.k_mu if plan is not None else session_kmu_c_per_mu(session_id, base_dir)
    return SessionSplatSource(session_id, iso, chamber, plan, n_raw, geom, k_mu)


def _load_timeslice_source(session_id: str, base_dir: str) -> SessionSplatSource | None:
    from ..common.timeslice_position_error import TIMESLICE_POSITION_ERROR_COLS

    from ..common.schema import resolve_column_name

    usecols = list(dict.fromkeys([
        *TIMESLICE_POSITION_ERROR_COLS, *TIMESLICE_SIGMA_COLS, *(c for _ic, c in _SCAN_DOSE_COLS),
    ]))
    iso_hooks = timeslice_position_table_hooks(REFERENCE_ISO)
    ch_hooks = timeslice_position_table_hooks(REFERENCE_CHAMBER)
    keys = (
        "ic1_x_iso", "ic1_y_iso", "ic2_x_iso", "ic2_y_iso",
        "ic1_x_ch", "ic1_y_ch", "ic2_x_ch", "ic2_y_ch",
        "ic1_sx", "ic1_sy", "ic2_sx", "ic2_sy",
        "ic1_nc", "ic2_nc", "spot_no",
    )

    def prepare(src, frames):
        iso_src = iso_hooks[0](src, frames)
        ch_src = ch_hooks[0](src, frames)
        sig_src = resolve_timeslice_sigma_source(frames[0].columns)
        if sig_src is None or (iso_src is None and ch_src is None):
            return None
        dose_cols = [resolve_column_name(frames[0].columns, c) for _ic, c in _SCAN_DOSE_COLS]
        spot_col = resolve_column_name(frames[0].columns, "spot_no")
        return iso_src, ch_src, sig_src, dose_cols, spot_col

    def extract(df, context):
        iso_src, ch_src, sig_src, dose_cols, spot_col = context
        charge = tuple(
            timeslice_charge_nc(df[c].to_numpy(dtype=float)) if c else np.full(len(df), np.nan)
            for c in dose_cols
        )
        spot = (
            pd.to_numeric(df[spot_col], errors="coerce").to_numpy() if spot_col
            else np.full(len(df), np.nan)
        )
        n = len(df)
        empty = tuple(np.full(n, np.nan) for _ in range(4))
        iso = iso_hooks[1](df, iso_src) if iso_src is not None else None
        ch = ch_hooks[1](df, ch_src) if ch_src is not None else None
        sig = frame_timeslice_sigma_arrays(df, sig_src)
        if iso is None:
            iso = empty
        if ch is None:
            ch = empty
        if sig is None:
            sig = empty
        if not any(np.isfinite(np.asarray(a)).any() for a in (*iso, *ch, *sig)):
            return None
        return (*iso, *ch, *sig, *charge, spot)

    table = load_energy_tagged_table(
        session_id,
        base_dir,
        usecols=usecols,
        prepare=prepare,
        extract=extract,
        keys=keys,
    )
    if table is None:
        return None
    beam_on = table.get("beam_on")
    if beam_on is not None:
        on = np.asarray(beam_on, dtype=bool)
        table = {k: (v[on] if isinstance(v, np.ndarray) and v.shape[0] == on.size else v)
                 for k, v in table.items()}
    energy = np.asarray(table["energy"], dtype=float)
    n = energy.size
    weight = np.ones(n, dtype=float)
    dose: dict[str, np.ndarray | None] = {}
    for ic in ("ic1", "ic2"):
        k = session_kmu_c_per_mu(session_id, base_dir, f"IC_{ic[-1]}")
        nc = np.asarray(table[f"{ic}_nc"], dtype=float)
        if np.nansum(np.abs(nc)) < _COULOMB_SESSION_MAX:
            nc = nc * 1e9
        dose[ic] = (
            nonnegative_layer_mu(nc / (k * 1e9), energy) if k and np.isfinite(nc).any() else None
        )
    charged = dose["ic1"] is not None
    if charged:
        # Slices that added no charge must weigh 0, not the count of 1.
        weight = dose["ic1"]

    def xy(key: str) -> np.ndarray:
        v = _sanitize_xy(table[key])
        return fill_from_spot(v, table["spot_no"], energy) if charged else v

    def sig(key: str) -> np.ndarray:
        v = _sanitize_sigma(table[key], scale=_SIGMA_STRIP_MM)
        return fill_from_spot(v, table["spot_no"], energy) if charged else v

    iso = PositionSigmaFrame(
        energy=energy,
        ic1_x=xy("ic1_x_iso"),
        ic1_y=xy("ic1_y_iso"),
        ic2_x=xy("ic2_x_iso"),
        ic2_y=xy("ic2_y_iso"),
        ic1_sx=sig("ic1_sx"),
        ic1_sy=sig("ic1_sy"),
        ic2_sx=sig("ic2_sx"),
        ic2_sy=sig("ic2_sy"),
        weight=weight,
        dose_ic1=dose["ic1"],
        dose_ic2=dose["ic2"],
    )
    chamber = PositionSigmaFrame(
        energy=energy,
        ic1_x=xy("ic1_x_ch"),
        ic1_y=xy("ic1_y_ch"),
        ic2_x=xy("ic2_x_ch"),
        ic2_y=xy("ic2_y_ch"),
        ic1_sx=iso.ic1_sx,
        ic1_sy=iso.ic1_sy,
        ic2_sx=iso.ic2_sx,
        ic2_sy=iso.ic2_sy,
        weight=weight,
        dose_ic1=dose["ic1"],
        dose_ic2=dose["ic2"],
    )
    if not np.isfinite(iso.ic1_x).any() and not np.isfinite(iso.ic2_x).any():
        iso = None
    if not np.isfinite(chamber.ic1_x).any() and not np.isfinite(chamber.ic2_x).any():
        chamber = None
    if iso is None and chamber is None:
        return None
    plan = _plan_from_input_map(session_id, base_dir)
    k_mu = plan.k_mu if plan is not None else session_kmu_c_per_mu(session_id, base_dir)
    return SessionSplatSource(
        session_id,
        iso,
        chamber,
        plan,
        n,
        load_session_map2map_geometry(session_id, base_dir),
        k_mu,
    )


def load_session_splat_source(
    session_id: str,
    base_dir: str,
    grain: str,
) -> SessionSplatSource | None:
    if grain == GRAIN_SPOT:
        return _load_spot_source(session_id, base_dir)
    if grain == GRAIN_TIMESLICE:
        return _load_timeslice_source(session_id, base_dir)
    return None


def load_splat_sessions(
    session_ids: Sequence[str],
    base_dir: str,
    grain: str,
) -> dict[str, SessionSplatSource]:
    out: dict[str, SessionSplatSource] = {}
    for sid in session_ids:
        src = load_session_splat_source(sid, base_dir, grain)
        if src is not None:
            out[sid] = src
    return out


def _axis_cloud(
    frame: PositionSigmaFrame,
    ic: str,
) -> SplatCloud:
    if ic == "ic1":
        return SplatCloud(
            x=frame.ic1_x, y=frame.ic1_y, sx=frame.ic1_sx, sy=frame.ic1_sy,
            energy=frame.energy, weight=frame.weight, dose=_frame_dose(frame, "ic1"),
        )
    return SplatCloud(
        x=frame.ic2_x, y=frame.ic2_y, sx=frame.ic2_sx, sy=frame.ic2_sy,
        energy=frame.energy, weight=frame.weight, dose=_frame_dose(frame, "ic2"),
    )


def _lerp_sigma(near: np.ndarray, far: np.ndarray, t: float) -> np.ndarray:
    """Interpolate IC2→IC1 sigma; if one side is missing, use the other."""
    a = np.asarray(near, dtype=float)
    b = np.asarray(far, dtype=float)
    out = (1.0 - t) * a + t * b
    miss = ~np.isfinite(out)
    if not miss.any():
        return out
    out = out.copy()
    use_a = miss & np.isfinite(a)
    use_b = miss & np.isfinite(b) & ~use_a
    out[use_a] = a[use_a]
    out[use_b] = b[use_b]
    return out


def _iso_ray_cloud(
    frame: PositionSigmaFrame,
    geom: Map2MapGeometry | None,
    plane: str = SIGMA_PLANE_CHAMBER,
) -> SplatCloud:
    off2x, off1x = ic_alignment_offsets(frame.ic2_x, frame.ic1_x)
    off2y, off1y = ic_alignment_offsets(frame.ic2_y, frame.ic1_y)
    p2x = frame.ic2_x - off2x
    p1x = frame.ic1_x - off1x
    p2y = frame.ic2_y - off2y
    p1y = frame.ic1_y - off1y
    aligned = PositionSigmaFrame(
        energy=frame.energy,
        ic1_x=p1x, ic1_y=p1y, ic2_x=p2x, ic2_y=p2y,
        ic1_sx=frame.ic1_sx, ic1_sy=frame.ic1_sy,
        ic2_sx=frame.ic2_sx, ic2_sy=frame.ic2_sy,
        weight=frame.weight,
        plan_x=frame.plan_x, plan_y=frame.plan_y,
    )
    z_iso = resolve_iso_z_mm(aligned, geom)
    x = iso_xy_from_ic_ray(p2x, p1x, z_iso)
    y = iso_xy_from_ic_ray(p2y, p1y, z_iso)
    t = (z_iso - IC2_Z_MM) / IC_SEP_MM if IC_SEP_MM else 1.0
    sx = _lerp_sigma(frame.ic2_sx, frame.ic1_sx, t)
    sy = _lerp_sigma(frame.ic2_sy, frame.ic1_sy, t)
    cloud = SplatCloud(
        x=x, y=y, sx=sx, sy=sy, energy=frame.energy, weight=frame.weight,
        dose=_frame_dose(frame, "iso"),
    )
    return _to_plane(cloud, geom, t, plane)


def measured_cloud(
    source: SessionSplatSource,
    xy_mode: str,
    base_dir: str = "",
    plane: str = SIGMA_PLANE_CHAMBER,
) -> SplatCloud | None:
    if xy_mode == XY_PLAN:
        cloud = source.plan
    elif xy_mode == XY_ISO_RAY:
        frame = source.chamber if source.chamber is not None else source.iso
        cloud = None if frame is None else _iso_ray_cloud(frame, source.geom, plane)
    else:
        frame = source.iso if source.iso is not None else source.chamber
        if frame is None or xy_mode not in (XY_IC1, XY_IC2):
            cloud = None
        else:
            t = 1.0 if xy_mode == XY_IC1 else 0.0
            cloud = _to_plane(_axis_cloud(frame, xy_mode), source.geom, t, plane)
    if cloud is None or source.k_mu is None or cloud.k_mu == source.k_mu:
        return cloud
    return replace(cloud, k_mu=source.k_mu)


def plan_cloud(
    source: SessionSplatSource,
    config: SplatConfig,
    reference: SessionSplatSource | None = None,
) -> SplatCloud | None:
    """The input-map plan with its spot σ from ``config.plan_sigma``, at ``config.sigma_plane``.

    Measured and Reference take the median σ of each energy layer from a session's own
    IC readings (the same XY as the view, IC1 when XY is Plan). Interlock keeps the
    devices.xml IC1 values, and so does a source with no usable σ.
    """
    plan = source.plan
    if plan is None:
        return None
    plane = config.sigma_plane
    if config.plan_sigma != PLAN_SIGMA_INTERLOCK:
        donor = reference if config.plan_sigma == PLAN_SIGMA_REFERENCE else source
        xy = XY_IC1 if config.xy_mode == XY_PLAN else config.xy_mode
        cloud = None if donor is None else measured_cloud(donor, xy, plane=plane)
        got = None if cloud is None else layer_sigmas(cloud, plan.energy)
        if got is not None:
            return replace(plan, sx=got[0], sy=got[1])
    return _to_plane(plan, source.geom, 1.0, plane)


def build_view_batches(
    sources: dict[str, SessionSplatSource],
    session_ids: Sequence[str],
    config: SplatConfig,
    base_dir: str,
) -> tuple[SplatBatch | None, SplatBatch | None, DepthAxis, int]:
    """Measured and plan spots of *session_ids*, each capped at ``config.splat_cap``, and the raw count."""
    measured_clouds: list[SplatCloud] = []
    plan_clouds: list[SplatCloud] = []
    n_raw = 0
    for sid in session_ids:
        source = sources.get(sid)
        if source is None:
            continue
        n_raw += source.n_raw
        plan = plan_cloud(source, config, sources.get(config.plan_sigma_ref))
        if config.xy_mode == XY_PLAN:
            measured = plan
        else:
            measured = measured_cloud(source, config.xy_mode, base_dir, config.sigma_plane)
            if config.overlay_plan and plan is not None:
                plan_clouds.append(plan)
        if measured is not None:
            measured_clouds.append(measured)
    axis = range_axis_for_medium(config.medium)

    def batch(clouds: list[SplatCloud]) -> SplatBatch | None:
        cloud = concat_clouds(clouds)
        if cloud is None:
            return None
        out = cloud_to_batch(apply_splat_cap(cloud, config.splat_cap), axis, config.smear_axis_units)
        return out if out.center.size else None

    return batch(measured_clouds), batch(plan_clouds), axis, n_raw
