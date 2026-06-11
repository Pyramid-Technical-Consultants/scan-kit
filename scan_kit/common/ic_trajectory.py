"""IC chamber geometry and beam deflection angle from paired IC readings."""

from __future__ import annotations

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
