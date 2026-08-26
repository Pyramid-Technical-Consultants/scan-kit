"""Tests for trajectory pivot / iso fit helpers."""

from __future__ import annotations

import numpy as np

from scan_kit.common.ic_trajectory import IC1_Z_MM, IC2_Z_MM, IC_SEP_MM
from scan_kit.common.trajectory_fits import (
    fit_iso_plane,
    fit_magnet_pivot,
    project_plan_to_z,
)


def _fan_positions(z_pivot: float, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IC2/IC1 lateral positions for rays from *z_pivot* with tan(theta) = angles."""
    ic2 = angles * (IC2_Z_MM - z_pivot)
    ic1 = angles * (IC1_Z_MM - z_pivot)
    return ic2, ic1


def test_fit_magnet_pivot_recovers_upstream_crossing() -> None:
    z_pivot = IC2_Z_MM - 1500.0
    angles = np.linspace(-0.03, 0.03, 60)
    ic2, ic1 = _fan_positions(z_pivot, angles)
    fit = fit_magnet_pivot(ic2, ic1)
    assert fit.is_valid
    np.testing.assert_allclose(fit.z_pivot, z_pivot, rtol=1e-4)
    np.testing.assert_allclose(fit.upstream_mm, IC2_Z_MM - z_pivot, rtol=1e-4)


def test_project_plan_to_z_scales_with_depth() -> None:
    plan = np.array([2.0, -1.0])
    z_pivot = IC2_Z_MM - 1000.0
    z_iso = IC2_Z_MM + 500.0
    z_mid = (z_pivot + z_iso) / 2.0
    projected = project_plan_to_z(plan, z_pivot, z_iso, z_mid)
    np.testing.assert_allclose(projected, plan * 0.5)


def test_fit_iso_plane_recovers_downstream_crossing() -> None:
    z_pivot = IC2_Z_MM - 1200.0
    z_iso = IC2_Z_MM + 800.0
    angles = np.linspace(-0.02, 0.02, 80)
    ic2, ic1 = _fan_positions(z_pivot, angles)
    plan = ic2 + (z_iso - IC2_Z_MM) * (ic1 - ic2) / IC_SEP_MM
    fit = fit_iso_plane(ic2, ic1, plan, z_pivot)
    assert fit.is_valid
    np.testing.assert_allclose(fit.z_iso, z_iso, rtol=1e-3)
