"""IC chamber geometry and beam deflection angle from paired IC readings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Plot origin at IC2 (first chamber along the downstream beam).
IC2_Z_MM = 0.0
IC1_Z_MM = 100.0
IC_SEP_MM = IC1_Z_MM - IC2_Z_MM

# G2 raw strip register range (see transform.py).
_G2_STRIP_VALID = (1.0, 128.0)


def ic_alignment_offsets(p2: np.ndarray, p1: np.ndarray) -> tuple[float, float]:
    """Rigid per-IC offset (median raw mm) = chamber alignment vs the beam axis.

    *p2* is upstream (IC2), *p1* downstream (IC1).  Subtracting these centres
    each chamber's cloud on-axis without changing slope (hence angle).
    """
    return float(np.nanmedian(p2)), float(np.nanmedian(p1))


@dataclass(frozen=True)
class IcFanConvergence:
    """Least-squares crossing point of the back-projected IC2→IC1 ray fan.

    Each spot's IC2→IC1 line, extended upstream, should cross the beam axis at
    the scan-magnet pivot.  ``position_mm`` is the lateral position of that best
    common crossing; once each chamber's alignment offset is removed it should
    sit on-axis (≈ 0), which is the sanity check that the fan converges at the
    origin.
    """

    z_pivot_mm: float  # mm downstream of IC2 (negative = upstream toward magnet)
    position_mm: float  # lateral crossing position (≈ 0 after alignment)

    @property
    def upstream_mm(self) -> float:
        """Distance upstream of IC2 to the pivot (positive = toward magnet)."""
        return IC2_Z_MM - self.z_pivot_mm

    @property
    def is_valid(self) -> bool:
        return np.isfinite(self.z_pivot_mm) and np.isfinite(self.position_mm)


def ic_fan_convergence(
    p2: np.ndarray,
    p1: np.ndarray,
    *,
    ic_sep_mm: float = IC_SEP_MM,
) -> IcFanConvergence:
    """Where the back-projected per-spot lines best converge (least squares).

    Pass **alignment-corrected** positions (offsets already subtracted).  The
    crossing *z* minimises the spread of lateral positions across spots; the
    crossing position is reported so callers can confirm the fan converges on
    the beam axis (≈ 0 mm).
    """
    p2 = np.asarray(p2, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    ok = np.isfinite(p2) & np.isfinite(p1)
    p2 = p2[ok]
    p1 = p1[ok]
    if p2.size < 2:
        return IcFanConvergence(float("nan"), float("nan"))

    slopes = (p1 - p2) / ic_sep_mm
    var_s = float(np.var(slopes))
    if var_s <= 0.0:
        return IcFanConvergence(float("nan"), float("nan"))

    z_pivot = -float(np.cov(p2, slopes, bias=True)[0, 1]) / var_s
    position = float(np.mean(p2) + z_pivot * float(np.mean(slopes)))
    return IcFanConvergence(z_pivot, position)


def beam_slopes(
    p2: np.ndarray,
    p1: np.ndarray,
    *,
    ic_sep_mm: float = IC_SEP_MM,
) -> np.ndarray:
    """Tangent of deflection angle from IC2→IC1 chamber positions (dimensionless)."""
    return (np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)) / ic_sep_mm


def beam_angles_mrad(
    p2: np.ndarray,
    p1: np.ndarray,
    *,
    ic_sep_mm: float = IC_SEP_MM,
) -> np.ndarray:
    """Deflection angle (mrad) from IC2→IC1 chamber positions in mm.

    ``np.arctan`` already returns radians, so milliradians is simply the angle
    in radians times 1000 (do *not* convert to degrees first).
    """
    slopes = beam_slopes(p2, p1, ic_sep_mm=ic_sep_mm)
    return np.arctan(slopes) * 1000.0


def aligned_beam_angles_mrad(
    ic2: np.ndarray,
    ic1: np.ndarray,
    *,
    ic2_offset: float,
    ic1_offset: float,
    ic_sep_mm: float = IC_SEP_MM,
) -> np.ndarray:
    """Deflection angle after subtracting session alignment offsets."""
    return beam_angles_mrad(
        np.asarray(ic2, dtype=float) - ic2_offset,
        np.asarray(ic1, dtype=float) - ic1_offset,
        ic_sep_mm=ic_sep_mm,
    )
