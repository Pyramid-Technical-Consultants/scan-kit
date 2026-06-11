"""Tests for IC beam trajectory helpers."""

from __future__ import annotations

import numpy as np

from scan_kit.common.ic_trajectory import (
    IC_SEP_MM,
    aligned_beam_angles_mrad,
    beam_angles_mrad,
    beam_slopes,
    ic_alignment_offsets,
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
