"""Proton depth physics for the dose volume: range, straggling, Bragg curve, scatter, gamma.

Everything here is a closed form or a one-time table per medium or energy
layer, so the fill stays one texture lookup per voxel.

- Stopping power: Bethe, no shell or density-effect corrections.
- CSDA range and range straggling: integrals of Bethe and Bohr over energy.
- Depth dose: Bortfeld (1997), with a local power law fitted to the Bethe range.
- Lateral spread: Fermi–Eyges with a Highland-style scattering power.
- Gamma index: global dose difference, distance to agreement, low-dose cutoff.

These are QA models for comparing measured against plan in a uniform medium.
They leave out tissue heterogeneity and the nuclear halo.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

# Bethe constant 4π N_A r_e² m_e c² (MeV cm²/mol) and rest energies (MeV).
_K_BETHE = 0.307075
_ME = 0.51099895
_MP = 938.27208816
# Energy spread of a spot, relative to its energy, when nothing better is known.
DEFAULT_ENERGY_SPREAD_PCT = 1.0
# Bortfeld: share of nuclear-interaction energy absorbed locally.
_NUCLEAR_LOCAL = 0.6
# Scattering constant for projected angles (MeV). Tuned so the end-of-range
# width in water is within ~5 % of Preston & Koehler over 100–230 MeV.
_E_SCATTER = 13.0
MEV_TO_GY_MM3 = 1.602176634e-7  # MeV deposited → Gy·mm³ at 1 g/cm³
_TABLE_E = np.geomspace(1.0, 400.0, 4000)
LAYER_NODES = 1024
# Energies are grouped to this step so a session's layers share tables.
LAYER_STEP_MEV = 0.1
MAX_LAYERS = 1024


@dataclass(frozen=True)
class Medium:
    key: str
    label: str
    z_over_a: float
    i_ev: float
    rho: float
    x0_g_cm2: float
    # ponytail: fixed non-elastic removal (cm²/g); ICRU 63 cross sections if depth dose looks off.
    nuclear_cm2_g: float


WATER = Medium("water", "water", 0.55509, 75.0, 1.0, 36.08, 0.012)
COPPER = Medium("copper", "copper", 0.45636, 322.0, 8.96, 12.86, 0.0074)
MEDIA = {m.key: m for m in (WATER, COPPER)}


def medium_for(key: str) -> Medium:
    return MEDIA.get(key, WATER)


def _beta2(e):
    g = 1.0 + np.asarray(e, dtype=float) / _MP
    return 1.0 - 1.0 / (g * g), g


def mass_stopping_power(medium: Medium, energy_mev) -> np.ndarray:
    """Bethe electronic mass stopping power (MeV cm²/g) for protons."""
    b2, g = _beta2(np.maximum(energy_mev, 0.5))
    ratio = _ME / _MP
    tmax = 2.0 * _ME * b2 * g * g / (1.0 + 2.0 * g * ratio + ratio * ratio)
    i_mev = medium.i_ev * 1e-6
    arg = 2.0 * _ME * b2 * g * g * tmax / (i_mev * i_mev)
    return _K_BETHE * medium.z_over_a / b2 * (0.5 * np.log(arg) - b2)


@dataclass(frozen=True)
class RangeTable:
    energy: np.ndarray
    range_g: np.ndarray
    straggle_g2: np.ndarray


@functools.cache
def range_table(key: str) -> RangeTable:
    """CSDA range and Bohr range-straggling variance from 1 to 400 MeV."""
    medium = medium_for(key)
    e = _TABLE_E
    s = mass_stopping_power(medium, e)
    # Residual range below the table: R = E / (p S) for a p ≈ 1.75 power law.
    r0 = e[0] / (1.75 * s[0])
    de = np.diff(e)
    inv = 1.0 / s
    rng = r0 + np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * de)])
    b2, _g = _beta2(e)
    omega = _K_BETHE * _ME * medium.z_over_a * (1.0 - 0.5 * b2) / (1.0 - b2)
    dvar = omega / s**3
    var = np.concatenate([[0.0], np.cumsum(0.5 * (dvar[1:] + dvar[:-1]) * de)])
    return RangeTable(e, rng, var)


def _interp_log(x, xp, fp):
    return np.interp(np.log(np.maximum(x, 1e-12)), np.log(xp), fp)


def csda_range_mm(medium: Medium, energy_mev) -> np.ndarray:
    t = range_table(medium.key)
    return _interp_log(np.maximum(energy_mev, t.energy[0]), t.energy, t.range_g) * 10.0 / medium.rho


def energy_at_range_mm(medium: Medium, range_mm) -> np.ndarray:
    """Residual energy of a proton with *range_mm* left to travel."""
    t = range_table(medium.key)
    r_g = np.asarray(range_mm, dtype=float) * medium.rho / 10.0
    return np.exp(np.interp(np.log(np.maximum(r_g, 1e-9)), np.log(t.range_g), np.log(t.energy)))


def straggle_mm(medium: Medium, energy_mev) -> np.ndarray:
    t = range_table(medium.key)
    var = _interp_log(np.maximum(energy_mev, t.energy[0]), t.energy, t.straggle_g2)
    return np.sqrt(np.maximum(var, 0.0)) * 10.0 / medium.rho


def depth_sigma_mm(medium: Medium, energy_mev, spread_pct: float) -> np.ndarray:
    """Range straggling and beam energy spread, added in quadrature."""
    e = np.maximum(np.asarray(energy_mev, dtype=float), 1.0)
    dr_de = 10.0 / (medium.rho * mass_stopping_power(medium, e))
    spread = dr_de * e * max(float(spread_pct), 0.0) / 100.0
    return np.sqrt(straggle_mm(medium, e) ** 2 + spread**2)


@functools.cache
def range_exponent(key: str) -> float:
    """``p`` in ``R = α E^p``, fitted over the therapy band."""
    t = range_table(key)
    band = (t.energy >= 20.0) & (t.energy <= 250.0)
    return float(np.polyfit(np.log(t.energy[band]), np.log(t.range_g[band]), 1)[0])


def _smoothed_power(nu: float, r, sigma: float) -> np.ndarray:
    """``max(r, 0)**nu`` convolved with a normal of width *sigma*."""
    from scipy.special import gamma as gamma_fn, pbdv

    r = np.asarray(r, dtype=float)
    zeta = np.clip(r / sigma, -40.0, None)
    out = np.empty_like(r)
    far = zeta > 30.0
    out[far] = np.power(r[far], nu)
    near = ~far
    d, _ = pbdv(-nu - 1.0, -zeta[near])
    out[near] = (
        sigma**nu * gamma_fn(nu + 1.0) * np.exp(-zeta[near] ** 2 / 4.0) * d / math.sqrt(2.0 * math.pi)
    )
    return out


def bragg_idd(medium: Medium, energy_mev: float, spread_pct: float, depth_mm) -> np.ndarray:
    """Bortfeld depth dose per proton (MeV/mm), integrated over the beam cross-section."""
    e0 = max(float(energy_mev), 1.0)
    p = range_exponent(medium.key)
    r0 = float(csda_range_mm(medium, e0)) / 10.0
    sigma = max(float(depth_sigma_mm(medium, e0, spread_pct)) / 10.0, 1e-4)
    alpha = r0 / e0**p
    beta = medium.nuclear_cm2_g * medium.rho
    r = r0 - np.asarray(depth_mm, dtype=float) / 10.0
    coef = beta * (1.0 + _NUCLEAR_LOCAL * p)
    val = _smoothed_power(1.0 / p - 1.0, r, sigma) + coef * _smoothed_power(1.0 / p, r, sigma)
    return val / (p * alpha ** (1.0 / p) * (1.0 + beta * r0)) / 10.0


def local_energy_fraction(medium: Medium, energy_mev: float) -> float:
    """Share of the beam energy Bortfeld deposits locally (the rest leaves as neutrals)."""
    p = range_exponent(medium.key)
    br = medium.nuclear_cm2_g * medium.rho * float(csda_range_mm(medium, energy_mev)) / 10.0
    return (1.0 + br * (1.0 + _NUCLEAR_LOCAL * p) / (1.0 + p)) / (1.0 + br)


def _pv(e):
    e = np.asarray(e, dtype=float)
    return e * (e + 2.0 * _MP) / (e + _MP)


def mcs_sigma_mm(medium: Medium, energy_mev: float, depth_mm) -> np.ndarray:
    """Fermi–Eyges projected lateral spread from multiple Coulomb scattering."""
    e0 = max(float(energy_mev), 1.0)
    r0 = float(csda_range_mm(medium, e0))
    s = np.linspace(0.0, r0, 2048)
    e = np.maximum(energy_at_range_mm(medium, r0 - s), 0.5)
    x0_mm = medium.x0_g_cm2 / medium.rho * 10.0
    t = (_E_SCATTER / _pv(e)) ** 2 / x0_mm
    ds = np.diff(s)

    def cum(f):
        return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * ds)])

    a0, a1, a2 = cum(t), cum(t * s), cum(t * s * s)
    var = np.maximum(s * s * a0 - 2.0 * s * a1 + a2, 0.0)
    z = np.clip(np.asarray(depth_mm, dtype=float), 0.0, r0)
    return np.sqrt(np.interp(z, s, var))


@dataclass(frozen=True)
class LayerTables:
    """Per-energy depth tables in scene millimetres (``z = −depth``).

    ``cdf[l]`` is the fraction of layer *l*'s energy deposited below each
    node, rising from 0 at ``zmin`` to 1 at the surface ``zmax = 0``.
    ``mcs[l]`` is the scatter width at each node.
    """

    energies: np.ndarray
    zmin: np.ndarray
    zmax: np.ndarray
    cdf: np.ndarray
    mcs: np.ndarray
    energy_dep: np.ndarray
    peak_per_mm: np.ndarray
    mcs_max: np.ndarray

    def layer_of(self, energy) -> np.ndarray:
        e = np.asarray(energy, dtype=float).reshape(-1)
        idx = np.searchsorted(self.energies, e)
        idx = np.clip(idx, 1, max(len(self.energies) - 1, 1))
        lo = np.clip(idx - 1, 0, len(self.energies) - 1)
        hi = np.clip(idx, 0, len(self.energies) - 1)
        pick_hi = np.abs(self.energies[hi] - e) < np.abs(self.energies[lo] - e)
        return np.where(pick_hi, hi, lo).astype(np.int64)


def layer_energies(energy) -> np.ndarray:
    e = np.asarray(energy, dtype=float).reshape(-1)
    e = e[np.isfinite(e) & (e > 1.0)]
    if e.size == 0:
        return np.array([100.0])
    step = LAYER_STEP_MEV
    layers = np.unique(np.round(e / step) * step)
    while layers.size > MAX_LAYERS:
        step *= 2.0
        layers = np.unique(np.round(e / step) * step)
    return layers


def build_layer_tables(
    medium: Medium, energy, spread_pct: float, nodes: int = LAYER_NODES, scatter: bool = True,
) -> LayerTables:
    layers = layer_energies(energy)
    n = int(nodes)
    cdf = np.zeros((layers.size, n), dtype=np.float32)
    mcs = np.zeros((layers.size, n), dtype=np.float32)
    zmin = np.zeros(layers.size)
    dep = np.zeros(layers.size)
    peak = np.zeros(layers.size)
    for i, e0 in enumerate(layers):
        r0 = float(csda_range_mm(medium, e0))
        sig = float(depth_sigma_mm(medium, e0, spread_pct))
        dmax = r0 + 5.0 * sig
        depth = np.linspace(dmax, 0.0, n)
        idd = bragg_idd(medium, e0, spread_pct, depth)
        step = dmax / (n - 1)
        acc = np.concatenate([[0.0], np.cumsum(0.5 * (idd[1:] + idd[:-1]) * step)])
        total = float(acc[-1])
        dep[i] = total
        cdf[i] = acc / max(total, 1e-30)
        peak[i] = float(np.max(idd)) / max(total, 1e-30)
        if scatter:
            mcs[i] = mcs_sigma_mm(medium, e0, depth)
        zmin[i] = -dmax
    return LayerTables(
        energies=layers,
        zmin=zmin,
        zmax=np.zeros(layers.size),
        cdf=cdf,
        mcs=mcs,
        energy_dep=dep,
        peak_per_mm=peak,
        mcs_max=mcs.max(axis=1).astype(float),
    )


@dataclass(frozen=True)
class LayerDoseKernel:
    """Depth dose from the layer tables; lateral width grows with depth."""

    tables: LayerTables

    def _row(self, energy) -> int:
        return int(self.tables.layer_of(np.atleast_1d(energy)[:1])[0])

    def _nodes(self, row: int) -> np.ndarray:
        t = self.tables
        return np.linspace(t.zmin[row], t.zmax[row], t.cdf.shape[1])

    def mass_fraction(self, energy, z_lo, z_hi):
        row = self._row(energy)
        z = self._nodes(row)
        c = self.tables.cdf[row].astype(float)
        return np.interp(z_hi, z, c) - np.interp(z_lo, z, c)

    def lateral_mm(self, energy, z):
        row = self._row(energy)
        return np.interp(z, self._nodes(row), self.tables.mcs[row].astype(float))

    def lateral_max_mm(self, energy) -> float:
        return float(self.tables.mcs_max[self._row(energy)])

    def z_span(self, energy) -> tuple[float, float]:
        row = self._row(energy)
        return float(self.tables.zmin[row]), float(self.tables.zmax[row])

    def span(self, centers, sigmas, energy) -> tuple[np.ndarray, np.ndarray]:
        """Centers and σ whose ±4σ box holds each spot's whole depth dose and scatter."""
        t = self.tables
        c = np.array(centers, dtype=float).reshape(-1, 3)
        s = np.array(sigmas, dtype=float).reshape(-1, 3)
        row = t.layer_of(np.asarray(energy, dtype=float).reshape(-1)[: c.shape[0]])
        c[:, 2] = 0.5 * (t.zmin[row] + t.zmax[row])
        s[:, 0:2] = np.hypot(s[:, 0:2], t.mcs_max[row][:, None])
        s[:, 2] = (t.zmax[row] - t.zmin[row]) / 8.0
        return c, s


def through_wet(energy, wet_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Energy left and primary protons kept after *wet_mm* of water-equivalent material.

    Protons that stop inside come back as energy 0 and transmission 0.
    ponytail: straggling picked up in the material is not carried into the phantom;
    it is under 1 mm for a few cm of material. Carry a per-layer spread if it ever matters.
    """
    e = np.maximum(np.asarray(energy, dtype=float), 1.0)
    r = csda_range_mm(WATER, e)
    left = r - float(wet_mm)
    through = left > 0.0
    e_out = np.where(through, energy_at_range_mm(WATER, np.maximum(left, 1e-6)), 0.0)
    # Bortfeld's primary fluence: Φ ∝ 1 + β (R − z), β per mm of water.
    beta = WATER.nuclear_cm2_g * WATER.rho / 10.0
    kept = np.where(through, (1.0 + beta * left) / (1.0 + beta * r), 0.0)
    return e_out, kept


def end_scatter_mm(medium: Medium, energy) -> np.ndarray:
    """Scatter width where each proton stops, per spot (one table per energy layer)."""
    e = np.asarray(energy, dtype=float).reshape(-1)
    nodes = layer_energies(e)
    width = [float(mcs_sigma_mm(medium, x, csda_range_mm(medium, x))) for x in nodes]
    return np.interp(e, nodes, width)


@functools.lru_cache(maxsize=8)
def _cached_kernel(
    key: str, layers: tuple[float, ...], spread_pct: float, scatter: bool,
) -> LayerDoseKernel:
    return LayerDoseKernel(
        build_layer_tables(medium_for(key), np.array(layers), spread_pct, scatter=scatter),
    )


def layer_kernel(key: str, energy, spread_pct: float, scatter: bool = True) -> LayerDoseKernel:
    """Depth-dose kernel for the energies present; reused while layers and σE are unchanged."""
    layers = tuple(float(e) for e in layer_energies(energy))
    return _cached_kernel(key, layers, round(float(spread_pct), 4), bool(scatter))


def dose_weights(kernel: LayerDoseKernel, protons, energy, medium: Medium) -> np.ndarray:
    """Gy·mm³ per spot: protons × energy deposited (MeV) converted for the medium's density."""
    t = kernel.tables
    dep = t.energy_dep[t.layer_of(np.asarray(energy, dtype=float).reshape(-1))]
    return np.asarray(protons, dtype=float).reshape(-1) * dep * MEV_TO_GY_MM3 / medium.rho


def effective_sigma_z(kernel: LayerDoseKernel, energy) -> np.ndarray:
    """Gaussian σ with the same peak height as each spot's depth dose (for color heuristics)."""
    t = kernel.tables
    peak = t.peak_per_mm[t.layer_of(np.asarray(energy, dtype=float).reshape(-1))]
    return 1.0 / (math.sqrt(2.0 * math.pi) * np.maximum(peak, 1e-9))


@dataclass(frozen=True)
class GammaCriteria:
    dose_pct: float = 3.0
    dta_mm: float = 2.0
    cutoff_pct: float = 10.0
    # γ above this is not searched for; the display tops out here too.
    cap: float = 2.0


def gamma_offsets(voxel_mm: float, criteria: GammaCriteria) -> np.ndarray:
    """Search offsets in voxel units, nearest first; ``w`` is (distance / DTA)²."""
    dta = max(float(criteria.dta_mm), 1e-3)
    # A whole fraction of the voxel keeps exact shifts on the grid; ≤ DTA/4 bounds the miss.
    step = float(voxel_mm) / math.ceil(float(voxel_mm) / (dta / 4.0))
    reach = criteria.cap * dta
    n = int(math.ceil(reach / step))
    ax = np.arange(-n, n + 1) * step
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    d2 = (gx**2 + gy**2 + gz**2) / dta**2
    keep = d2 <= criteria.cap**2
    out = np.column_stack([gx[keep], gy[keep], gz[keep]]) / float(voxel_mm)
    w = d2[keep]
    order = np.argsort(w, kind="stable")
    return np.column_stack([out[order], w[order]]).astype(np.float32)


def gamma_index(ref, evl, voxel_mm: float, criteria: GammaCriteria) -> tuple[np.ndarray, int, int]:
    """Global 3D gamma of *ref* points searched in *evl*. Arrays are ``(nz, ny, nx)``.

    Returns the γ volume (0 below the cutoff), points passing, and points evaluated.
    """
    from scipy import ndimage

    ref = np.asarray(ref, dtype=np.float64)
    evl = np.asarray(evl, dtype=np.float64)
    norm = float(np.max(evl)) if evl.size else 0.0
    out = np.zeros(ref.shape, dtype=np.float32)
    if norm <= 0.0:
        return out, 0, 0
    dd = criteria.dose_pct / 100.0 * norm
    mask = (ref > 0.0) & (ref >= criteria.cutoff_pct / 100.0 * norm)
    best = np.full(ref.shape, criteria.cap**2)
    # ponytail: one shifted copy per offset; fine for tests and the no-compute fallback.
    for ox, oy, oz, w in gamma_offsets(voxel_mm, criteria):
        if w >= best[mask].max(initial=0.0):
            break
        moved = ndimage.shift(evl, (-oz, -oy, -ox), order=1, mode="constant", cval=0.0)
        np.minimum(best, w + ((moved - ref) / dd) ** 2, out=best)
    gam = np.sqrt(best)
    out[mask] = gam[mask]
    evaluated = int(mask.sum())
    return out, int((gam[mask] <= 1.0).sum()), evaluated
