"""Load IC Gaussians and map them through a pluggable depth axis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

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
from .gaussian_splat_catalog import (
    GRAIN_SPOT,
    GRAIN_TIMESLICE,
    MEDIUM_AIR,
    MEDIUM_WATER,
    XY_IC1,
    XY_IC2,
    XY_ISO_RAY,
    XY_PLAN,
    SplatConfig,
)

_FALLBACK_SIGMA_MM = 4.0
_ABS_INVALID_MM = abs(INVALID_POSITION_MM) * 0.9


class DepthAxis(Protocol):
    """Maps energy (MeV) to scene-mm depth (range in a medium)."""

    axis_label: str
    smear_label: str

    def z_scene_mm(self, energy_mev: np.ndarray) -> np.ndarray: ...

    def sigma_z_scene_mm(
        self, energy_mev: np.ndarray, smear_axis_units: float,
    ) -> np.ndarray: ...


# Bortfeld 1997: R_cm = 0.0022 E^{1.77}. Air uses water range / ρ_air.
# ponytail: density scale only; I-value CSDA tables if a third medium shows up.
_BORTFELD_P = 1.77
_BORTFELD_ALPHA_WATER_MM = 0.022
_RHO_AIR_G_CM3 = 0.001204


@dataclass(frozen=True)
class LinearEnergyAxis:
    """Linear MeV→mm (tests / protocol). Live view uses :class:`RangeAxis`."""

    mm_per_mev: float
    e_min: float = 0.0
    axis_label: str = "Energy (MeV)"
    smear_label: str = "MeV"

    def z_scene_mm(self, energy_mev: np.ndarray) -> np.ndarray:
        return (np.asarray(energy_mev, dtype=float) - self.e_min) * self.mm_per_mev

    def sigma_z_scene_mm(
        self, energy_mev: np.ndarray, smear_axis_units: float,
    ) -> np.ndarray:
        thickness = abs(float(smear_axis_units) * self.mm_per_mev)
        return np.full(np.shape(energy_mev), max(thickness, 1e-6), dtype=float)


@dataclass(frozen=True)
class RangeAxis:
    """CSDA-like proton range: ``R = α E^p`` (mm, MeV). Higher E → deeper."""

    medium: str
    alpha_mm: float
    p: float = _BORTFELD_P
    axis_label: str = "Range in water (mm)"
    smear_label: str = "MeV"

    def z_scene_mm(self, energy_mev: np.ndarray) -> np.ndarray:
        e = np.maximum(np.asarray(energy_mev, dtype=float), 0.0)
        return self.alpha_mm * np.power(e, self.p)

    def sigma_z_scene_mm(
        self, energy_mev: np.ndarray, smear_axis_units: float,
    ) -> np.ndarray:
        e = np.maximum(np.asarray(energy_mev, dtype=float), 1.0)
        dr_de = self.alpha_mm * self.p * np.power(e, self.p - 1.0)
        return np.maximum(np.abs(dr_de * float(smear_axis_units)), 1e-6)


def range_axis_for_medium(medium: str) -> RangeAxis:
    if medium == MEDIUM_AIR:
        return RangeAxis(
            medium=MEDIUM_AIR,
            alpha_mm=_BORTFELD_ALPHA_WATER_MM / _RHO_AIR_G_CM3,
            axis_label="Range in air (mm)",
        )
    return RangeAxis(
        medium=MEDIUM_WATER,
        alpha_mm=_BORTFELD_ALPHA_WATER_MM,
        axis_label="Range in water (mm)",
    )


@dataclass(frozen=True)
class SplatCloud:
    x: np.ndarray
    y: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    energy: np.ndarray
    weight: np.ndarray


@dataclass(frozen=True)
class SplatBatch:
    center: np.ndarray
    sigma: np.ndarray
    weight: np.ndarray
    rgb: np.ndarray
    energy_mev: np.ndarray


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


@dataclass(frozen=True)
class SessionSplatSource:
    session_id: str
    iso: PositionSigmaFrame | None
    chamber: PositionSigmaFrame | None
    plan: SplatCloud | None
    n_raw: int
    geom: Map2MapGeometry | None = None


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


def default_mm_per_mev(energy: np.ndarray, xy_span: float) -> float:
    e = np.asarray(energy, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return 1.0
    e_span = float(np.ptp(e))
    if e_span <= 0.0 or not np.isfinite(xy_span) or xy_span <= 0.0:
        return 1.0
    return xy_span / e_span


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


def energy_rgb(
    energy: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    from matplotlib import cm

    e = np.asarray(energy, dtype=float)
    finite = np.isfinite(e)
    norm = np.full(e.shape, 0.5, dtype=float)
    if finite.any():
        lo = float(np.min(e[finite])) if vmin is None else float(vmin)
        hi = float(np.max(e[finite])) if vmax is None else float(vmax)
        if hi > lo:
            norm[finite] = np.clip((e[finite] - lo) / (hi - lo), 0.0, 1.0)
    rgba = np.asarray(cm.viridis(norm), dtype=np.float32)
    rgb = rgba[:, :3].copy()
    rgb[~finite] = 0.0
    return rgb


def cloud_to_batch(
    cloud: SplatCloud,
    axis: DepthAxis,
    smear_axis_units: float,
    rgb: np.ndarray,
) -> SplatBatch:
    z = axis.z_scene_mm(cloud.energy)
    sz = axis.sigma_z_scene_mm(cloud.energy, smear_axis_units)
    mask = _finite_mask(cloud.x, cloud.y, cloud.sx, cloud.sy, cloud.energy, z, sz)
    center = np.column_stack([cloud.x[mask], cloud.y[mask], z[mask]]).astype(np.float32)
    sigma = np.column_stack([cloud.sx[mask], cloud.sy[mask], sz[mask]]).astype(np.float32)
    weight = np.asarray(cloud.weight[mask], dtype=np.float32)
    color = np.asarray(rgb[mask], dtype=np.float32)
    return SplatBatch(
        center=center,
        sigma=sigma,
        weight=weight,
        rgb=color,
        energy_mev=np.asarray(cloud.energy[mask], dtype=np.float32),
    )


def autoscale_axis(clouds: Sequence[SplatCloud], mm_per_mev: float) -> LinearEnergyAxis:
    energies = [c.energy for c in clouds if c.energy.size]
    if not energies:
        return LinearEnergyAxis(mm_per_mev=1.0)
    energy = np.concatenate(energies)
    finite_e = energy[np.isfinite(energy)]
    e_min = float(np.min(finite_e)) if finite_e.size else 0.0
    if mm_per_mev > 0.0:
        return LinearEnergyAxis(mm_per_mev=mm_per_mev, e_min=e_min)
    xs = np.concatenate([c.x for c in clouds])
    ys = np.concatenate([c.y for c in clouds])
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not ok.any():
        span = 100.0
    else:
        span = float(max(np.ptp(xs[ok]), np.ptp(ys[ok]), 1.0))
    return LinearEnergyAxis(mm_per_mev=default_mm_per_mev(energy, span), e_min=e_min)


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


def iso_sigma_scale(geom: Map2MapGeometry | None, t: float) -> float:
    # ponytail: linear interpolate chamber mag; magnet-pivot drift is the upgrade.
    if geom is None:
        return 1.0
    m1 = geom.mag_factor("IC_1_X")
    m2 = geom.mag_factor("IC_2_X")
    if m1 is None and m2 is None:
        return 1.0
    a = float(m2) if m2 is not None else 1.0
    b = float(m1) if m1 is not None else 1.0
    return (1.0 - t) * a + t * b


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
    return SplatCloud(x=x, y=y, sx=sx, sy=sy, energy=energy, weight=weight)


def _sigma_columns(spot_columns) -> dict[str, str]:
    found: dict[str, str] = {}
    for label, ic, axis in IC_SIGMA_LABELS:
        col = resolve_spot_sigma_column(spot_columns, ic, axis)
        if col is not None:
            found[label] = col
    return found


def _frame_from_position_data(
    data: dict,
    sigma_cols: dict[str, str],
    *,
    sigma_scale: float,
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
    )


def _load_spot_frame(
    session_id: str,
    base_dir: str,
    *,
    raw: bool,
    sigma_cols: dict[str, str],
) -> PositionSigmaFrame | None:
    extra_spot = list(sigma_cols.values())

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
    return _frame_from_position_data(data, sigma_cols, sigma_scale=2.0)


def _load_spot_source(session_id: str, base_dir: str) -> SessionSplatSource | None:
    _input_map, spot_data = load_session_raw(session_id, base_dir=base_dir)
    if spot_data is None:
        return None
    sigma_cols = _sigma_columns(spot_data.columns)
    iso = _load_spot_frame(session_id, base_dir, raw=False, sigma_cols=sigma_cols)
    chamber = _load_spot_frame(session_id, base_dir, raw=True, sigma_cols=sigma_cols)
    if iso is None and chamber is None:
        return None
    plan = _plan_from_input_map(session_id, base_dir)
    n_raw = int((iso or chamber).energy.size)
    geom = load_session_map2map_geometry(session_id, base_dir)
    return SessionSplatSource(session_id, iso, chamber, plan, n_raw, geom)


def _load_timeslice_source(session_id: str, base_dir: str) -> SessionSplatSource | None:
    from ..common.timeslice_position_error import TIMESLICE_POSITION_ERROR_COLS

    usecols = list(dict.fromkeys([*TIMESLICE_POSITION_ERROR_COLS, *TIMESLICE_SIGMA_COLS]))
    iso_hooks = timeslice_position_table_hooks(REFERENCE_ISO)
    ch_hooks = timeslice_position_table_hooks(REFERENCE_CHAMBER)
    keys = (
        "ic1_x_iso", "ic1_y_iso", "ic2_x_iso", "ic2_y_iso",
        "ic1_x_ch", "ic1_y_ch", "ic2_x_ch", "ic2_y_ch",
        "ic1_sx", "ic1_sy", "ic2_sx", "ic2_sy",
    )

    def prepare(src, frames):
        iso_src = iso_hooks[0](src, frames)
        ch_src = ch_hooks[0](src, frames)
        sig_src = resolve_timeslice_sigma_source(frames[0].columns)
        if sig_src is None or (iso_src is None and ch_src is None):
            return None
        return iso_src, ch_src, sig_src

    def extract(df, context):
        iso_src, ch_src, sig_src = context
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
        return (*iso, *ch, *sig)

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
    iso = PositionSigmaFrame(
        energy=energy,
        ic1_x=_sanitize_xy(table["ic1_x_iso"]),
        ic1_y=_sanitize_xy(table["ic1_y_iso"]),
        ic2_x=_sanitize_xy(table["ic2_x_iso"]),
        ic2_y=_sanitize_xy(table["ic2_y_iso"]),
        ic1_sx=_sanitize_sigma(table["ic1_sx"]),
        ic1_sy=_sanitize_sigma(table["ic1_sy"]),
        ic2_sx=_sanitize_sigma(table["ic2_sx"]),
        ic2_sy=_sanitize_sigma(table["ic2_sy"]),
        weight=weight,
    )
    chamber = PositionSigmaFrame(
        energy=energy,
        ic1_x=_sanitize_xy(table["ic1_x_ch"]),
        ic1_y=_sanitize_xy(table["ic1_y_ch"]),
        ic2_x=_sanitize_xy(table["ic2_x_ch"]),
        ic2_y=_sanitize_xy(table["ic2_y_ch"]),
        ic1_sx=iso.ic1_sx,
        ic1_sy=iso.ic1_sy,
        ic2_sx=iso.ic2_sx,
        ic2_sy=iso.ic2_sy,
        weight=weight,
    )
    if not np.isfinite(iso.ic1_x).any() and not np.isfinite(iso.ic2_x).any():
        iso = None
    if not np.isfinite(chamber.ic1_x).any() and not np.isfinite(chamber.ic2_x).any():
        chamber = None
    if iso is None and chamber is None:
        return None
    return SessionSplatSource(
        session_id,
        iso,
        chamber,
        _plan_from_input_map(session_id, base_dir),
        n,
        load_session_map2map_geometry(session_id, base_dir),
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
            energy=frame.energy, weight=frame.weight,
        )
    return SplatCloud(
        x=frame.ic2_x, y=frame.ic2_y, sx=frame.ic2_sx, sy=frame.ic2_sy,
        energy=frame.energy, weight=frame.weight,
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
    scale = iso_sigma_scale(geom, t)
    sx = _lerp_sigma(frame.ic2_sx, frame.ic1_sx, t) * scale
    sy = _lerp_sigma(frame.ic2_sy, frame.ic1_sy, t) * scale
    return SplatCloud(
        x=x, y=y, sx=sx, sy=sy, energy=frame.energy, weight=frame.weight,
    )


def measured_cloud(
    source: SessionSplatSource,
    xy_mode: str,
    base_dir: str = "",
) -> SplatCloud | None:
    if xy_mode == XY_PLAN:
        return source.plan
    if xy_mode == XY_ISO_RAY:
        frame = source.chamber if source.chamber is not None else source.iso
        if frame is None:
            return None
        return _iso_ray_cloud(frame, source.geom)
    frame = source.iso if source.iso is not None else source.chamber
    if frame is None:
        return None
    if xy_mode == XY_IC1:
        return _axis_cloud(frame, "ic1")
    if xy_mode == XY_IC2:
        return _axis_cloud(frame, "ic2")
    return None


def build_view_batches(
    sources: dict[str, SessionSplatSource],
    session_ids: Sequence[str],
    config: SplatConfig,
    base_dir: str,
    *,
    plan_rgb: tuple[float, float, float],
) -> tuple[SplatBatch | None, SplatBatch | None, DepthAxis, int, int]:
    measured_clouds: list[SplatCloud] = []
    plan_clouds: list[SplatCloud] = []
    n_raw = 0
    for sid in session_ids:
        source = sources.get(sid)
        if source is None:
            continue
        n_raw += source.n_raw
        measured = measured_cloud(source, config.xy_mode, base_dir)
        if measured is not None:
            measured_clouds.append(measured)
        if config.overlay_plan and config.xy_mode != XY_PLAN and source.plan is not None:
            plan_clouds.append(source.plan)
    measured_one = concat_clouds(measured_clouds)
    if measured_one is not None:
        measured_one = apply_splat_cap(measured_one, config.splat_cap)
    plan_one = concat_clouds(plan_clouds)
    if plan_one is not None:
        plan_one = apply_splat_cap(plan_one, config.splat_cap)
    measured_clouds = [measured_one] if measured_one is not None else []
    plan_clouds = [plan_one] if plan_one is not None else []
    clouds = [*measured_clouds, *plan_clouds]
    axis = range_axis_for_medium(config.medium)
    if not clouds:
        return None, None, axis, n_raw, 0
    e_lo, e_hi = 0.0, 1.0
    if measured_one is not None:
        finite = measured_one.energy[np.isfinite(measured_one.energy)]
        if finite.size:
            e_lo = float(np.min(finite))
            e_hi = float(np.max(finite))

    def _merge(parts: list[SplatCloud], rgb_fn) -> SplatBatch | None:
        batches = [
            cloud_to_batch(cloud, axis, config.smear_axis_units, rgb_fn(cloud))
            for cloud in parts
        ]
        batches = [b for b in batches if b.center.size]
        if not batches:
            return None
        return SplatBatch(
            center=np.concatenate([b.center for b in batches], axis=0),
            sigma=np.concatenate([b.sigma for b in batches], axis=0),
            weight=np.concatenate([b.weight for b in batches], axis=0),
            rgb=np.concatenate([b.rgb for b in batches], axis=0),
            energy_mev=np.concatenate([b.energy_mev for b in batches], axis=0),
        )

    measured_batch = _merge(
        measured_clouds,
        lambda c: energy_rgb(c.energy, vmin=e_lo, vmax=e_hi),
    )
    plan_batch = _merge(
        plan_clouds,
        lambda c: np.tile(np.asarray(plan_rgb, dtype=np.float32), (c.x.size, 1)),
    )
    n_used = 0
    if measured_batch is not None:
        n_used += int(measured_batch.center.shape[0])
    if plan_batch is not None:
        n_used += int(plan_batch.center.shape[0])
    return measured_batch, plan_batch, axis, n_raw, n_used
