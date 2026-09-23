"""Voxel dose grid (1 mm by default). Spots deposit by the analytic Gaussian integral.

The Z factor goes through :class:`DepthKernel`: a Gaussian stopping peak
here, or the Bragg-curve tables in :mod:`dose_volume_physics`.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

VOXEL_MM = 1.0
MIN_VOXEL_MM = 0.25
MAX_VOXEL_MM = 10.0
# Beam-plane tiles and compute bricks are counted in cells, not millimetres.
TILE_CELLS = 16
MAX_CELLS = 512
SIGMA_CUT = 4.0
# A run of this many millimetres at the typical cell peak is optical depth ~1 at gain 1.
VIEW_DEPTH_MM = 8.0
DOSE_FLOOR = 0.02
# Values under this fraction of the color reach fade into the view background.
HOT_RGB = (0.90, 0.16, 0.14)
COLD_RGB = (0.16, 0.40, 0.90)

_SQRT2 = math.sqrt(2.0)


class DepthKernel(Protocol):
    """Fraction of a spot's weight whose depth falls in ``[z_lo, z_hi]``."""

    def mass_fraction(self, energy, z_lo, z_hi): ...


@dataclass(frozen=True)
class GaussianSmearKernel:
    """Z mass is the range-axis energy smear. Today's only depth kernel."""

    axis: object
    smear: float

    def mass_fraction(self, energy, z_lo, z_hi):
        e = np.atleast_1d(np.asarray(energy, dtype=float))
        mu = float(np.asarray(self.axis.z_scene_mm(e[:1]), dtype=float).reshape(-1)[0])
        sig = float(np.asarray(
            self.axis.sigma_z_scene_mm(e[:1], self.smear), dtype=float,
        ).reshape(-1)[0])
        return normal_mass(mu, sig, z_lo, z_hi)


@dataclass(frozen=True)
class DoseGrid:
    """Min corner of voxel (0,0,0), cell counts ``(nx, ny, nz)``, and cell edge in mm."""

    origin: np.ndarray
    shape: tuple[int, int, int]
    cropped: bool
    voxel: float = VOXEL_MM

    @property
    def extent_mm(self) -> np.ndarray:
        return np.asarray(self.shape, dtype=float) * float(self.voxel)


def _erf(x):
    """Abramowitz & Stegun 7.1.26. Same coefficients as the compute shader."""
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
          - 0.284496736) * t + 0.254829592) * t
    return np.sign(x) * (1.0 - p * np.exp(-ax * ax))


def normal_mass(mu, sigma, lo, hi):
    """Normal CDF difference across ``[lo, hi]``. ``mu`` and ``sigma`` broadcast."""
    sig = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    mu = np.asarray(mu, dtype=float)
    z1 = (np.asarray(hi, dtype=float) - mu) / (sig * _SQRT2)
    z0 = (np.asarray(lo, dtype=float) - mu) / (sig * _SQRT2)
    return 0.5 * (_erf(z1) - _erf(z0))


def dose_field_weights(dose, fallback) -> np.ndarray:
    """Delivered MU where it is finite and positive, otherwise *fallback*."""
    base = np.asarray(fallback, dtype=np.float64).reshape(-1)
    if dose is None:
        return base.astype(np.float32)
    got = np.asarray(dose, dtype=np.float64).reshape(-1)
    if got.size != base.size:
        return base.astype(np.float32)
    use = np.isfinite(got) & (got > 0.0)
    return np.where(use, got, base).astype(np.float32)


def clamp_voxel(voxel_mm: float) -> float:
    v = float(voxel_mm)
    if not math.isfinite(v):
        return VOXEL_MM
    return min(max(v, MIN_VOXEL_MM), MAX_VOXEL_MM)


def dose_grid(centers, sigmas, voxel_mm: float = VOXEL_MM, z_floor: float | None = None) -> DoseGrid:
    """Grid on the spots plus 4 sigma, snapped to whole voxels.

    *z_floor* is the back face of a finite phantom: nothing is deposited below it.
    An axis longer than :data:`MAX_CELLS` is cropped to that many cells about
    the middle of the spot extent.
    """
    v = clamp_voxel(voxel_mm)
    c = np.asarray(centers, dtype=float).reshape(-1, 3)
    s = np.maximum(np.asarray(sigmas, dtype=float).reshape(-1, 3), 1e-6)
    if c.size == 0:
        return DoseGrid(np.zeros(3, dtype=float), (1, 1, 1), False, v)
    ok = np.isfinite(c).all(axis=1) & np.isfinite(s).all(axis=1)
    if not ok.any():
        return DoseGrid(np.zeros(3, dtype=float), (1, 1, 1), False, v)
    c = c[ok]
    s = s[ok]
    lo = np.floor(np.min(c - SIGMA_CUT * s, axis=0) / v)
    hi = np.ceil(np.max(c + SIGMA_CUT * s, axis=0) / v)
    if z_floor is not None:
        # Round inward so no voxel reaches past the back face.
        lo[2] = max(lo[2], math.ceil(float(z_floor) / v))
        hi[2] = max(hi[2], lo[2] + 1)
    shape = np.maximum(np.rint(hi - lo).astype(int), 1)
    cropped = False
    for i in range(3):
        if int(shape[i]) > MAX_CELLS:
            cropped = True
            mid = 0.5 * (float(lo[i]) + float(hi[i]))
            lo[i] = math.floor(mid - MAX_CELLS / 2.0)
            shape[i] = MAX_CELLS
    return DoseGrid(
        (lo * v).astype(float), (int(shape[0]), int(shape[1]), int(shape[2])), cropped, v,
    )


def _index_span(
    center: float, sigma: float, origin: float, n_cells: int, voxel: float,
) -> tuple[int, int]:
    sig = max(float(sigma), 1e-6)
    i0 = int(math.floor((center - SIGMA_CUT * sig - origin) / voxel))
    i1 = int(math.ceil((center + SIGMA_CUT * sig - origin) / voxel))
    return max(0, i0), min(int(n_cells), max(i0, i1))


def deposit_gaussians(centers, sigmas, weights, energy, kernel: DepthKernel, grid: DoseGrid) -> np.ndarray:
    """MU per cell, shape ``(nz, ny, nx)`` so it uploads as a 3D texture.

    XY is the separable normal integral. Z is ``kernel.mass_fraction``.
    Divide by ``grid.voxel ** 3`` for MU / mm³.

    A kernel with ``lateral_mm`` widens *sigmas* in quadrature at each depth,
    and one with ``z_span`` sets the depth range instead of ``center ± 4σ``.
    """
    nx, ny, nz = grid.shape
    vol = np.zeros((nz, ny, nx), dtype=np.float64)
    c = np.asarray(centers, dtype=float).reshape(-1, 3)
    s = np.asarray(sigmas, dtype=float).reshape(-1, 3)
    w = np.asarray(weights, dtype=float).reshape(-1)
    e = np.asarray(energy, dtype=float).reshape(-1)
    origin = np.asarray(grid.origin, dtype=float).reshape(3)
    v = float(grid.voxel)
    n = min(c.shape[0], s.shape[0], w.shape[0], e.shape[0])
    lateral = getattr(kernel, "lateral_mm", None)
    z_span = getattr(kernel, "z_span", None)
    for i in range(n):
        if not np.isfinite(w[i]) or w[i] == 0.0 or not np.isfinite(c[i]).all():
            continue
        grow = kernel.lateral_max_mm(e[i]) if lateral else 0.0
        ix0, ix1 = _index_span(c[i, 0], math.hypot(s[i, 0], grow), origin[0], nx, v)
        iy0, iy1 = _index_span(c[i, 1], math.hypot(s[i, 1], grow), origin[1], ny, v)
        if z_span:
            z0, z1 = z_span(e[i])
            iz0 = max(0, int(math.floor((z0 - origin[2]) / v)))
            iz1 = min(nz, int(math.ceil((z1 - origin[2]) / v)))
        else:
            iz0, iz1 = _index_span(c[i, 2], s[i, 2], origin[2], nz, v)
        if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
            continue
        x_lo = origin[0] + np.arange(ix0, ix1) * v
        y_lo = origin[1] + np.arange(iy0, iy1) * v
        z_lo = origin[2] + np.arange(iz0, iz1) * v
        fz = np.asarray(kernel.mass_fraction(e[i], z_lo, z_lo + v), dtype=float).reshape(-1)
        if lateral:
            mcs = np.asarray(lateral(e[i], z_lo + 0.5 * v), dtype=float).reshape(-1, 1)
            sx = np.sqrt(s[i, 0] ** 2 + mcs**2)
            sy = np.sqrt(s[i, 1] ** 2 + mcs**2)
            fx = normal_mass(c[i, 0], sx, x_lo[None, :], x_lo[None, :] + v)
            fy = normal_mass(c[i, 1], sy, y_lo[None, :], y_lo[None, :] + v)
            vol[iz0:iz1, iy0:iy1, ix0:ix1] += (
                w[i] * fz[:, None, None] * fy[:, :, None] * fx[:, None, :]
            )
            continue
        fx = normal_mass(c[i, 0], s[i, 0], x_lo, x_lo + v)
        fy = normal_mass(c[i, 1], s[i, 1], y_lo, y_lo + v)
        vol[iz0:iz1, iy0:iy1, ix0:ix1] += (
            w[i] * fz[:, None, None] * fy[None, :, None] * fx[None, None, :]
        )
    return vol.astype(np.float32)


def tile_shape(grid: DoseGrid) -> tuple[int, int]:
    nx, ny, _nz = grid.shape
    return (nx + TILE_CELLS - 1) // TILE_CELLS, (ny + TILE_CELLS - 1) // TILE_CELLS


def tile_slot_capacity(centers, sigmas, grid: DoseGrid) -> int:
    """Upper bound on spot-id slots. Sizes the GPU buffer; the GPU writes the lists."""
    c = np.asarray(centers, dtype=float).reshape(-1, 3)
    s = np.asarray(sigmas, dtype=float).reshape(-1, 3)
    n = min(c.shape[0], s.shape[0])
    c = c[:n]
    s = s[:n]
    ok = np.isfinite(c).all(axis=1)
    if not ok.any():
        return 1
    ntx, nty = tile_shape(grid)
    origin = np.asarray(grid.origin, dtype=float).reshape(3)
    tile_mm = TILE_CELLS * float(grid.voxel)
    tx = _tile_span_counts(c[ok, 0], s[ok, 0], origin[0], ntx, tile_mm)
    ty = _tile_span_counts(c[ok, 1], s[ok, 1], origin[1], nty, tile_mm)
    return max(int(np.sum(tx * ty)), 1)


def _tile_span_counts(coord, sigma, origin: float, n_tiles: int, tile_mm: float) -> np.ndarray:
    sig = np.maximum(np.nan_to_num(sigma, nan=1e-6), 1e-6)
    t0 = np.floor((coord - SIGMA_CUT * sig - origin) / tile_mm)
    t1 = np.floor((coord + SIGMA_CUT * sig - origin) / tile_mm)
    t0 = np.clip(t0, 0, n_tiles - 1)
    t1 = np.clip(t1, 0, n_tiles - 1)
    return (t1 - t0 + 1).astype(np.int64)


def typical_ray(weights, sigmas) -> float:
    """95th percentile of the line integral through each spot center.

    Taken along the axis that collects the most (the narrowest sigma).
    At 1 mm cells this is MU/mm² when the weights are MU.
    """
    w = np.abs(np.asarray(weights, dtype=float).reshape(-1))
    s = np.maximum(np.asarray(sigmas, dtype=float).reshape(-1, 3), 1e-6)
    if w.size == 0 or s.shape[0] == 0:
        return 1.0
    n = min(w.size, s.shape[0])
    w = w[:n]
    s = s[:n]
    best = np.zeros(n)
    for axis in range(3):
        perp = [a for a in range(3) if a != axis]
        ray = w.copy()
        for a in perp:
            ray = ray * normal_mass(0.0, s[:, a], -0.5, 0.5)
        best = np.maximum(best, ray)
    best = best[np.isfinite(best) & (best > 0.0)]
    if best.size == 0:
        return 1.0
    return max(float(np.percentile(best, 95)), 1e-12)


def typical_cell(weights, sigmas) -> float:
    """95th percentile of the MU in the 1 mm cell centered on each spot."""
    w = np.abs(np.asarray(weights, dtype=float).reshape(-1))
    s = np.maximum(np.asarray(sigmas, dtype=float).reshape(-1, 3), 1e-6)
    if w.size == 0 or s.shape[0] == 0:
        return 1.0
    n = min(w.size, s.shape[0])
    cell = w[:n]
    for axis in range(3):
        cell = cell * normal_mass(0.0, s[:n, axis], -0.5, 0.5)
    cell = cell[np.isfinite(cell) & (cell > 0.0)]
    if cell.size == 0:
        return 1.0
    return max(float(np.percentile(cell, 95)), 1e-12)


GAMMA_CMAP = "scan_gamma"


@functools.cache
def _colormap(name: str):
    """A matplotlib colormap; the gamma map is built here since matplotlib has none."""
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    if name != GAMMA_CMAP:
        return colormaps[name]
    # γ 0–2 on the bar: calm teal up to the pass line at 1, then a hard step to hot.
    return LinearSegmentedColormap.from_list(name, [
        (0.0, "#141a26"), (0.25, "#1f4e5f"), (0.499, "#6fc2b0"),
        (0.5, "#f5a524"), (0.75, "#e5484d"), (1.0, "#ff4fd8"),
    ], N=512)


def colormap_samples(name: str, n: int = 256) -> np.ndarray:
    """RGB samples of a matplotlib colormap, ``t`` from 0 to 1. Shape ``(n, 3)``."""
    t = np.linspace(0.0, 1.0, int(n))
    rgba = np.asarray(_colormap(name)(t), dtype=np.float32)
    return np.ascontiguousarray(rgba[:, :3])


def zero_rgb(name: str, lo: float, hi: float) -> tuple[float, float, float]:
    """Colormap color at 0 on a ``lo``–``hi`` scale, clamped to the nearer end."""
    t = 0.0 if not hi > lo else min(max(-float(lo) / (float(hi) - float(lo)), 0.0), 1.0)
    return tuple(float(c) for c in _colormap(name)(t)[:3])


def ink_rgb(bg) -> tuple[float, float, float]:
    """Light text on a dark background, dark text on a light one."""
    r, g, b = (float(c) for c in bg[:3])
    return (0.79, 0.82, 0.85) if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else (0.13, 0.14, 0.16)


def residual_error_alpha(mag, scale: float = 1.0):
    """Hermite smoothstep from 0 at no error to 1 at ``scale``."""
    hi = max(float(scale), 1e-12)
    x = np.clip(np.asarray(mag, dtype=np.float64) / hi, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def residual_signed_rgba(signed, *, scale: float = 1.0, zero: str = "transparent"):
    """Hot/cold legend ramp. Agreement is transparent or white."""
    signed = np.asarray(signed, dtype=np.float64)
    w = residual_error_alpha(np.abs(signed), scale)
    hot = signed >= 0.0
    rgb = np.empty(signed.shape + (3,), dtype=np.float64)
    rgb[hot] = HOT_RGB
    rgb[~hot] = COLD_RGB
    if zero == "white":
        rgb = (1.0 - w[..., None]) + w[..., None] * rgb
        alpha = np.ones(signed.shape, dtype=np.float64)
    else:
        alpha = w
    return np.concatenate([rgb, alpha[..., None]], axis=-1)
