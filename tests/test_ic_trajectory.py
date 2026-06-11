"""Tests for IC beam trajectory helpers."""

from __future__ import annotations

import numpy as np

from scan_kit.common.ic_trajectory import (
    IC2_Z_MM,
    IC_SEP_MM,
    aligned_beam_angles_mrad,
    beam_angles_mrad,
    beam_slopes,
    ic_alignment_offsets,
    ic_fan_convergence,
)


def test_beam_slope_from_chamber_positions() -> None:
    p2 = np.array([0.0, 0.0])
    p1 = np.array([1.0, -2.0])
    np.testing.assert_allclose(beam_slopes(p2, p1), [1.0 / IC_SEP_MM, -2.0 / IC_SEP_MM])


def test_alignment_offsets_center_parallel_beam() -> None:
    ic2 = np.linspace(-5.0, 5.0, 50)
    ic1 = ic2 + 5.0
    off2, off1 = ic_alignment_offsets(ic2, ic1)
    aligned = aligned_beam_angles_mrad(ic2, ic1, ic2_offset=off2, ic1_offset=off1)
    assert float(np.nanstd(aligned)) < 1e-6


def test_fan_convergence_recovers_pivot_on_axis() -> None:
    # Rays fan out from a pivot 2000 mm upstream of IC2, on-axis (position 0).
    z_pivot = IC2_Z_MM - 2000.0
    angles = np.linspace(-0.02, 0.02, 41)  # tan(theta) per spot
    # Each ray: position(z) = tan(theta) * (z - z_pivot).
    ic2 = angles * (IC2_Z_MM - z_pivot)
    ic1 = angles * (IC2_Z_MM + IC_SEP_MM - z_pivot)
    conv = ic_fan_convergence(ic2, ic1)
    assert conv.is_valid
    np.testing.assert_allclose(conv.z_pivot_mm, z_pivot, rtol=1e-6)
    assert abs(conv.position_mm) < 1e-6  # converges on-axis


def test_fan_convergence_offset_chambers_shift_off_axis() -> None:
    # Same fan but both chambers carry a relative offset (IC misalignment):
    # before alignment the fan no longer crosses the axis at 0.
    z_pivot = IC2_Z_MM - 2000.0
    angles = np.linspace(-0.02, 0.02, 41)
    ic2 = angles * (IC2_Z_MM - z_pivot) + 64.5
    ic1 = angles * (IC2_Z_MM + IC_SEP_MM - z_pivot) + 64.8
    raw = ic_fan_convergence(ic2, ic1)
    assert abs(raw.position_mm) > 1.0  # offset chambers => off-axis crossing
    off2, off1 = ic_alignment_offsets(ic2, ic1)
    aligned = ic_fan_convergence(ic2 - off2, ic1 - off1)
    assert abs(aligned.position_mm) < 1e-6  # alignment restores convergence at 0
