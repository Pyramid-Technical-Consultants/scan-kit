"""Upstream magnet pivot and downstream iso-plane fits for IC beam trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ic_trajectory import IC1_Z_MM, IC2_Z_MM, IC_SEP_MM

# Cap spots used for pairwise crossover (n^2 pairs); fixed seed for reproducibility.
_PAIRWISE_MAX_SPOTS = 400
_PAIRWISE_MIN_INTERSECTIONS = 20
_PAIRWISE_RNG_SEED = 0
_ISO_MIN_ESTIMATES = 20
_SLOPE_EPS = 1e-9


@dataclass(frozen=True)
class MagnetFit:
    """Upstream crossover of per-spot IC2–IC1 lines for one session axis."""

    z_pivot: float
    upstream_mm: float
    upstream_sigma_mm: float

    @property
    def is_valid(self) -> bool:
        return (
            np.isfinite(self.z_pivot)
            and np.isfinite(self.upstream_mm)
            and self.z_pivot < IC2_Z_MM
        )


@dataclass(frozen=True)
class IsoFit:
    """Downstream iso plane where measured trajectories match plan nominals."""

    z_iso: float
    downstream_mm: float
    downstream_sigma_mm: float

    @property
    def is_valid(self) -> bool:
        return np.isfinite(self.z_iso) and self.z_iso > IC2_Z_MM


def pairwise_upstream_crossings(p2: np.ndarray, slopes: np.ndarray) -> np.ndarray:
    """Z downstream from IC2 where pairs of spot lines intersect, upstream only."""
    n = p2.size
    z_vals: list[np.ndarray] = []
    for i in range(n - 1):
        dm = slopes[i] - slopes[i + 1 :]
        valid = np.abs(dm) > _SLOPE_EPS
        if not np.any(valid):
            continue
        z_ij = IC2_Z_MM + (p2[i + 1 :] - p2[i])[valid] / dm[valid]
        z_vals.append(z_ij[z_ij < IC2_Z_MM])
    if not z_vals:
        return np.array([], dtype=float)
    return np.concatenate(z_vals)


def fit_magnet_pivot(p2: np.ndarray, p1: np.ndarray) -> MagnetFit:
    """Median upstream crossover z from pairwise intersections of spot lines."""
    nan = MagnetFit(float("nan"), float("nan"), float("nan"))
    if p2.size < 2:
        return nan

    slopes = (p1 - p2) / IC_SEP_MM
    n = p2.size
    if n > _PAIRWISE_MAX_SPOTS:
        rng = np.random.default_rng(_PAIRWISE_RNG_SEED)
        pick = np.sort(rng.choice(n, size=_PAIRWISE_MAX_SPOTS, replace=False))
        p2 = p2[pick]
        slopes = slopes[pick]

    z_cross = pairwise_upstream_crossings(p2, slopes)
    if z_cross.size < _PAIRWISE_MIN_INTERSECTIONS:
        return nan

    z_pivot = float(np.median(z_cross))
    mad = float(np.median(np.abs(z_cross - z_pivot)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(z_cross, ddof=1))

    return MagnetFit(
        z_pivot,
        IC2_Z_MM - z_pivot,
        float(sigma) if np.isfinite(sigma) else float("nan"),
    )


def per_spot_iso_estimates(
    p2: np.ndarray,
    p1: np.ndarray,
    plan_p: np.ndarray,
) -> np.ndarray:
    """Z downstream from IC2 where each measured line crosses its plan nominal."""
    slopes = (p1 - p2) / IC_SEP_MM
    valid = (
        np.isfinite(p2)
        & np.isfinite(plan_p)
        & np.isfinite(slopes)
        & (np.abs(slopes) > _SLOPE_EPS)
    )
    z_iso = IC2_Z_MM + (plan_p[valid] - p2[valid]) / slopes[valid]
    return z_iso[np.isfinite(z_iso) & (z_iso > IC2_Z_MM)]


def fit_iso_plane(
    p2: np.ndarray,
    p1: np.ndarray,
    plan_p: np.ndarray,
    z_pivot: float,
) -> IsoFit:
    """Median downstream z where each measured line crosses its plan nominal."""
    nan = IsoFit(float("nan"), float("nan"), float("nan"))
    if p2.size < 2 or not np.isfinite(z_pivot):
        return nan

    ok = np.isfinite(p2) & np.isfinite(p1) & np.isfinite(plan_p)
    p2 = p2[ok]
    p1 = p1[ok]
    plan_p = plan_p[ok]
    if p2.size < 2:
        return nan

    z_spot = per_spot_iso_estimates(p2, p1, plan_p)
    if z_spot.size < _ISO_MIN_ESTIMATES:
        return nan

    n = z_spot.size
    if n > _PAIRWISE_MAX_SPOTS:
        rng = np.random.default_rng(_PAIRWISE_RNG_SEED)
        z_spot = np.sort(rng.choice(z_spot, size=_PAIRWISE_MAX_SPOTS, replace=False))

    z_iso = float(np.median(z_spot))
    mad = float(np.median(np.abs(z_spot - z_iso)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(z_spot, ddof=1))

    return IsoFit(
        z_iso,
        z_iso - IC2_Z_MM,
        float(sigma) if np.isfinite(sigma) else float("nan"),
    )


def project_plan_to_z(
    plan_p: np.ndarray,
    z_pivot: float,
    z_iso: float,
    z: float,
) -> np.ndarray:
    """Project iso-center plan position to lateral coordinate at *z*."""
    return plan_p * (z - z_pivot) / (z_iso - z_pivot)


def combined_pivot_z(magnet_x: MagnetFit, magnet_y: MagnetFit) -> float:
    """Best single pivot depth for 3D overlays (median of axis fits)."""
    vals = [
        magnet_x.z_pivot,
        magnet_y.z_pivot,
    ]
    ok = [v for v in vals if np.isfinite(v)]
    if not ok:
        return float("nan")
    return float(np.median(ok))


def combined_iso_z(iso_x: IsoFit | None, iso_y: IsoFit | None) -> float:
    vals: list[float] = []
    if iso_x is not None and iso_x.is_valid:
        vals.append(iso_x.z_iso)
    if iso_y is not None and iso_y.is_valid:
        vals.append(iso_y.z_iso)
    if not vals:
        return float("nan")
    return float(np.median(vals))
