"""Geometry for independent X/Y scanning dipoles (e.g. D2-650 double-dipole assembly).

Each scan axis is a straight deflection dipole: the beam enters on-axis, bends through
a circular arc in one plane, and exits at angle ``θ`` where ``sin θ ≈ L_eff / ρ`` for
small angles (``ρ = p/(qB)``, ``L_eff`` the effective magnetic length).  The *virtual
pivot* — where back-projected IC spot fans converge — sits near ``L_eff/2`` upstream
of each magnet's effective field center along the beam axis.

The D2-650-IE mounts two such dipoles for orthogonal X and Y steering.  They are
not co-located in general, so each axis has its own upstream convergence depth
``z_pivot`` and downstream iso crossing ``z_iso``, inferred separately from IC2/IC1
measurements (see ``trajectory_fits``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .ic_trajectory import IC1_Z_MM, IC2_Z_MM, IC_SEP_MM
from .trajectory_fits import project_plan_to_z

if TYPE_CHECKING:
    from .trajectory_fits import IsoFit, MagnetFit


def dipole_virtual_pivot_upstream_mm(effective_length_mm: float) -> float:
    """Hard-edge dipole: virtual pivot is ``L_eff / 2`` upstream of magnet center."""
    return effective_length_mm / 2.0


def beam_lateral_mm(
    p2: np.ndarray,
    p1: np.ndarray,
    z_mm: float | np.ndarray,
    *,
    ic_sep_mm: float = IC_SEP_MM,
) -> np.ndarray:
    """Lateral position along a measured IC2→IC1 ray at downstream *z* (mm from IC2)."""
    p2 = np.asarray(p2, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    slope = (p1 - p2) / ic_sep_mm
    z = np.asarray(z_mm, dtype=float)
    return p2 + slope * (z - IC2_Z_MM)


def axis_z_pivot(magnet: MagnetFit | None) -> float:
    if magnet is not None and magnet.is_valid:
        return magnet.z_pivot
    return float("nan")


def axis_z_iso(iso: IsoFit | None) -> float:
    if iso is not None and iso.is_valid:
        return iso.z_iso
    return float("nan")


def segment_z_bounds(
    magnet_x: MagnetFit | None,
    magnet_y: MagnetFit | None,
    iso_x: IsoFit | None,
    iso_y: IsoFit | None,
    *,
    extend_upstream_mm: float,
    extend_downstream_mm: float,
) -> tuple[float, float]:
    """Upstream/downstream clip range spanning both dipole pivots and iso planes."""
    z_start = IC2_Z_MM - extend_upstream_mm
    z_end = IC1_Z_MM + extend_downstream_mm

    pivots = [
        v for v in (axis_z_pivot(magnet_x), axis_z_pivot(magnet_y)) if np.isfinite(v)
    ]
    isos = [
        v for v in (axis_z_iso(iso_x), axis_z_iso(iso_y)) if np.isfinite(v)
    ]
    if pivots:
        z_start = min(pivots)
    if isos:
        z_end = max(isos)
    if z_end < z_start:
        z_end = z_start
    return z_start, z_end


def plan_lateral_at_z(
    plan_mm: np.ndarray,
    z_pivot: float,
    z_iso: float,
    z_mm: float,
) -> np.ndarray:
    """Plan nominal lateral position at *z* for one scan axis."""
    if not np.isfinite(z_pivot) or not np.isfinite(z_iso) or abs(z_iso - z_pivot) < 1e-3:
        return np.full(plan_mm.shape, np.nan, dtype=float)
    return project_plan_to_z(plan_mm, z_pivot, z_iso, z_mm)
