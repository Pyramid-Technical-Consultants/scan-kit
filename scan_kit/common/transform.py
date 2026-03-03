"""Coordinate transformation utilities for scan-kit."""

import numpy as np

# Standard IC coordinate mapping: raw 1-128 -> mm range
IC_RAW_MIN = 1
IC_RAW_MAX = 128
IC_MM_MIN = -128
IC_MM_MAX = 128

# IC1 X: 1-128 -> -128 to 128
IC1_X_MAP = (IC_RAW_MIN, IC_RAW_MAX, IC_MM_MIN, IC_MM_MAX)
# IC1 Y: 1-128 -> 128 to -128 (Y inverted)
IC1_Y_MAP = (IC_RAW_MIN, IC_RAW_MAX, IC_MM_MAX, IC_MM_MIN)
# IC2 X: 1-128 -> 128 to -128
IC2_X_MAP = (IC_RAW_MIN, IC_RAW_MAX, IC_MM_MAX, IC_MM_MIN)
# IC2 Y: 1-128 -> -128 to 128
IC2_Y_MAP = (IC_RAW_MIN, IC_RAW_MAX, IC_MM_MIN, IC_MM_MAX)


def remap(x, in_min, in_max, out_min, out_max):
    """Linear coordinate remapping from input range to output range."""
    x = np.asarray(x)
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
