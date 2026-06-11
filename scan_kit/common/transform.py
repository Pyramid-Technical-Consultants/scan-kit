"""Coordinate transformation utilities for scan-kit."""

import numpy as np

# Standard IC coordinate mapping: raw 1-128 -> mm range
IC_RAW_MIN = 1
IC_RAW_MAX = 128
IC_MM_MIN = -128
IC_MM_MAX = 128
IC_STRIP_CENTER = (IC_RAW_MIN + IC_RAW_MAX) / 2.0
IC_RAW_TO_MM = (IC_MM_MAX - IC_MM_MIN) / (IC_RAW_MAX - IC_RAW_MIN)

# G3 (and legacy) per-IC strip orientation — IC2 X/Y may be inverted vs IC1.
IC1_X_MAP = (IC_RAW_MIN, IC_RAW_MAX, IC_MM_MIN, IC_MM_MAX)
IC1_Y_MAP = (IC_RAW_MIN, IC_RAW_MAX, IC_MM_MAX, IC_MM_MIN)
IC2_X_MAP = (IC_RAW_MIN, IC_RAW_MAX, IC_MM_MAX, IC_MM_MIN)
IC2_Y_MAP = (IC_RAW_MIN, IC_RAW_MAX, IC_MM_MIN, IC_MM_MAX)


def remap(x, in_min, in_max, out_min, out_max):
    """Linear coordinate remapping from input range to output range."""
    x = np.asarray(x)
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


# G3 strip detector: 2 mm pitch, 128 channels (~256 mm wide), central strip 64.5.
G3_STRIP_CENTER = 64.5
G3_STRIP_PITCH_MM = 2.0


def remap_g3_raw(x):
    """G3 raw strip channel → mm (2 mm pitch, channel 64.5 → 0 mm).

    IC1 uses this forward sense; IC2 is mounted rotated 180° so it uses
    :func:`remap_g3_raw_reversed` (decided once per session).
    """
    return (np.asarray(x, dtype=float) - G3_STRIP_CENTER) * G3_STRIP_PITCH_MM


def remap_g3_raw_reversed(x):
    """G3 strip channel → mm for the reversed (180°-rotated) chamber sense."""
    return (G3_STRIP_CENTER - np.asarray(x, dtype=float)) * G3_STRIP_PITCH_MM


def remap_g2_raw(x):
    """G2 raw strip index → mm (register 64.5 → 0, 0→127 increasing).

    IC1 always uses this forward sense.  IC2 may use :func:`remap_g2_raw_reversed`
    when its strips run 127→0 (see :func:`g2_ic2_mm`).
    """
    return (np.asarray(x, dtype=float) - IC_STRIP_CENTER) * IC_RAW_TO_MM


def remap_g2_raw_reversed(x):
    """G2 IC2 when strip index runs 127→0 — still maps register 64.5 → 0 mm."""
    return (IC_STRIP_CENTER - np.asarray(x, dtype=float)) * IC_RAW_TO_MM


def g2_ic2_mm(ic1_mm, raw_ic2) -> np.ndarray:
    """G2 IC2 mm aligned to IC1, picking forward vs reversed strip direction.

    IC1 mm is the anchor (64.5 → 0).  Only IC2 sign is adjusted so per-spot
    |IC1 − IC2| is minimised; IC1 is never negated.
    """
    ic1 = np.asarray(ic1_mm, dtype=float)
    ic2_fwd = remap_g2_raw(raw_ic2)
    err_fwd = float(np.nanmedian(np.abs(ic1 - ic2_fwd)))
    err_rev = float(np.nanmedian(np.abs(ic1 + ic2_fwd)))
    if err_rev < err_fwd:
        return remap_g2_raw_reversed(raw_ic2)
    return ic2_fwd
