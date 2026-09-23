"""Tests for the dose-volume grid, loaders, and ray-march smoke."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from scan_kit.common.ic_trajectory import IC1_Z_MM, IC2_Z_MM, IC_SEP_MM
from scan_kit.views.dose_volume_catalog import (
    AGREE_TRANSPARENT,
    ERROR_ABSOLUTE,
    SplatConfig,
    WEIGHT_MU,
    WEIGHT_PROTONS,
    XY_IC1,
    XY_IC2,
    XY_ISO_RAY,
    XY_PLAN,
)
from scan_kit.views.dose_volume_data import (
    LinearEnergyAxis,
    PositionSigmaFrame,
    SessionSplatSource,
    SplatCloud,
    air_mass_stopping_mev_cm2_g,
    apply_splat_cap,
    cloud_to_batch,
    color_scalar,
    concat_clouds,
    default_mm_per_mev,
    energy_rgb,
    iso_xy_from_ic_ray,
    measured_cloud,
    measured_dose_columns,
    parse_kmu_c_per_mu,
    protons_from_charge_c,
    protons_from_mu,
    range_axis_for_medium,
)
from scan_kit.views.dose_volume_fill import (
    COLD_RGB,
    HOT_RGB,
    DoseGrid,
    dose_field_weights,
    dose_grid,
    deposit_gaussians,
    normal_mass,
    residual_error_alpha,
    residual_signed_rgba,
)
from scan_kit.views.dose_volume_vispy import apply_gantry, axis_guide_points


def test_iso_xy_from_ic_ray_matches_slope() -> None:
    p2 = np.array([0.0, 10.0])
    p1 = np.array([10.0, 20.0])
    z_iso = IC2_Z_MM + 2.0 * IC_SEP_MM
    got = iso_xy_from_ic_ray(p2, p1, z_iso)
    # At 2× IC separation the ray continues the same step beyond IC1.
    np.testing.assert_allclose(got, np.array([20.0, 30.0]))
    mid = iso_xy_from_ic_ray(p2, p1, (IC1_Z_MM + IC2_Z_MM) / 2.0)
    np.testing.assert_allclose(mid, np.array([5.0, 15.0]))


def test_linear_energy_axis_maps_energy_and_smear() -> None:
    axis = LinearEnergyAxis(mm_per_mev=2.0, e_min=70.0)
    z = axis.z_scene_mm(np.array([70.0, 80.0]))
    np.testing.assert_allclose(z, [0.0, 20.0])
    sz = axis.sigma_z_scene_mm(np.array([70.0, 80.0]), smear_axis_units=0.5)
    np.testing.assert_allclose(sz, [1.0, 1.0])
    assert axis.axis_label == "Energy (MeV)"
    assert axis.smear_label == "MeV"


def test_depth_axis_protocol_accepts_wet_stub() -> None:
    @dataclass(frozen=True)
    class WetStub:
        axis_label: str = "WET (mm)"
        smear_label: str = "mm"

        def z_scene_mm(self, energy_mev: np.ndarray) -> np.ndarray:
            return np.asarray(energy_mev, dtype=float) * 0.3

        def sigma_z_scene_mm(
            self, energy_mev: np.ndarray, smear_axis_units: float,
        ) -> np.ndarray:
            return np.full(np.shape(energy_mev), smear_axis_units, dtype=float)

    cloud = SplatCloud(
        x=np.array([1.0]),
        y=np.array([2.0]),
        sx=np.array([3.0]),
        sy=np.array([4.0]),
        energy=np.array([100.0]),
        weight=np.array([1.0]),
    )
    batch = cloud_to_batch(
        cloud, WetStub(), smear_axis_units=2.0, rgb=np.array([[1.0, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(batch.center[0], [1.0, 2.0, 30.0])
    np.testing.assert_allclose(batch.sigma[0], [3.0, 4.0, 2.0])


def test_cloud_to_batch_drops_nan_rows() -> None:
    cloud = SplatCloud(
        x=np.array([1.0, np.nan, 3.0]),
        y=np.array([1.0, 2.0, 3.0]),
        sx=np.array([2.0, 2.0, 2.0]),
        sy=np.array([2.0, 2.0, 2.0]),
        energy=np.array([70.0, 80.0, 90.0]),
        weight=np.array([1.0, 1.0, 1.0]),
    )
    batch = cloud_to_batch(
        cloud, LinearEnergyAxis(1.0), 0.5, np.ones((3, 3)),
    )
    assert batch.center.shape == (2, 3)
    np.testing.assert_allclose(batch.center[:, 0], [1.0, 3.0])


def test_concat_clouds_then_cap_is_global() -> None:
    def _cloud(x0: int, n: int) -> SplatCloud:
        return SplatCloud(
            x=np.arange(x0, x0 + n, dtype=float),
            y=np.zeros(n),
            sx=np.ones(n),
            sy=np.ones(n),
            energy=np.full(n, 70.0),
            weight=np.ones(n),
        )

    stacked = concat_clouds([_cloud(0, 6), _cloud(6, 6)])
    assert stacked is not None
    assert stacked.x.size == 12
    capped = apply_splat_cap(stacked, 4)
    assert capped.x.size <= 4


def test_apply_splat_cap_uses_stride() -> None:
    n = 10
    cloud = SplatCloud(
        x=np.arange(n, dtype=float),
        y=np.zeros(n),
        sx=np.ones(n),
        sy=np.ones(n),
        energy=np.linspace(70.0, 80.0, n),
        weight=np.ones(n),
    )
    capped = apply_splat_cap(cloud, 4)
    assert capped.x.size <= 4
    np.testing.assert_array_equal(capped.x, cloud.x[::3])


def test_range_axis_deeper_for_higher_energy() -> None:
    water = range_axis_for_medium("water")
    z = water.z_scene_mm(np.array([70.0, 100.0, 230.0]))
    # Beam from +Z (sky): higher energy is more negative (bottom at gantry 0°).
    assert z[0] > z[1] > z[2]
    np.testing.assert_allclose(z[1], -76.3, atol=1.5)
    copper = range_axis_for_medium("copper")
    z_cu = copper.z_scene_mm(np.array([100.0]))
    np.testing.assert_allclose(z_cu[0] / z[1], 1.0 / 8.96, rtol=1e-3)
    smear = water.sigma_z_scene_mm(np.array([100.0]), 1.0)
    # dR/dE = p R / E at 100 MeV ≈ 1.77 * 76.3 / 100 mm per MeV.
    np.testing.assert_allclose(smear[0], 1.77 * abs(z[1]) / 100.0, rtol=1e-5)
    assert water.depth_sign == -1
    assert "water" in water.axis_label.lower()
    assert "copper" in copper.axis_label.lower()


def test_default_mm_per_mev_matches_xy_span() -> None:
    energy = np.array([70.0, 90.0])
    assert default_mm_per_mev(energy, 40.0) == pytest.approx(2.0)


class _ZAtEnergy:
    """Depth kernel whose mean is the energy argument and whose sigma is fixed."""

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def mass_fraction(self, energy, z_lo, z_hi):
        return normal_mass(energy, self.sigma, z_lo, z_hi)


def _spot_volume(center, sigma, weight, *, origin, shape):
    grid = DoseGrid(np.asarray(origin, dtype=float), shape, False)
    return deposit_gaussians(
        np.asarray(center, dtype=float).reshape(1, 3),
        np.full((1, 3), sigma),
        np.asarray([weight], dtype=float),
        np.asarray([center[2]], dtype=float),
        _ZAtEnergy(sigma),
        grid,
    )


def test_typical_ray_is_the_central_line_integral() -> None:
    from scan_kit.views.dose_volume_fill import normal_mass, typical_ray

    sigma = 4.0
    weight = 8.0
    dphi = float(normal_mass(0.0, sigma, -0.5, 0.5))
    got = typical_ray([weight], [[sigma, sigma, sigma]])
    assert got == pytest.approx(weight * dphi * dphi, rel=1e-6)


def test_xy_switch_keeps_the_camera() -> None:
    from scan_kit.views.dose_volume_vispy import keep_camera

    assert not keep_camera(False, None, 90.0)
    assert keep_camera(True, 90.0, 90.0)
    assert not keep_camera(True, 90.0, 45.0)


def test_volume_frame_centers_the_grid() -> None:
    from scan_kit.views.dose_volume_vispy import apply_gantry, volume_corners, volume_frame

    origin = np.array([-10.0, -20.0, -100.0])
    shape = np.array([20.0, 40.0, 60.0])
    corners = volume_corners(origin, shape)
    mapped = apply_gantry(corners, 90.0)
    _lo, _hi, center = volume_frame(mapped)
    beam_center = (origin + shape / 2.0).reshape(1, 3)
    np.testing.assert_allclose(center, apply_gantry(beam_center, 90.0)[0], atol=1e-9)
    # A label stuck on one face shifts the framed center off the grid.
    labeled = np.vstack([mapped, mapped[0] + np.array([80.0, 0.0, 0.0])])
    _lo2, _hi2, shifted = volume_frame(labeled)
    assert abs(shifted[0] - center[0]) > 10.0


def test_one_spot_integrates_to_its_mu() -> None:
    vol = _spot_volume((0.0, 0.0, 0.0), 2.0, 3.0, origin=(-8.0, -8.0, -8.0), shape=(16, 16, 16))
    assert vol.sum() == pytest.approx(3.0, rel=1e-3)


def test_equal_volumes_cancel() -> None:
    a = _spot_volume((1.0, -2.0, 4.0), 1.5, 1.0, origin=(-8.0, -12.0, -4.0), shape=(20, 20, 16))
    b = _spot_volume((1.0, -2.0, 4.0), 1.5, 1.0, origin=(-8.0, -12.0, -4.0), shape=(20, 20, 16))
    assert np.max(np.abs(a - b)) == 0.0


def test_shifted_spot_is_a_dipole() -> None:
    origin = (-16.0, -8.0, -8.0)
    shape = (40, 16, 16)
    meas = _spot_volume((6.0, 0.0, 0.0), 2.0, 1.0, origin=origin, shape=shape)
    plan = _spot_volume((0.0, 0.0, 0.0), 2.0, 1.0, origin=origin, shape=shape)
    delta = meas - plan
    # +X side of the shift is extra measured dose. −X side is missing dose.
    plus = delta[:, :, 22:]
    minus = delta[:, :, :18]
    assert plus.sum() > 0.2
    assert minus.sum() < -0.2


def test_tail_outside_four_sigma_is_empty() -> None:
    vol = _spot_volume((0.0, 0.0, 0.0), 2.0, 1.0, origin=(-20.0, -20.0, -20.0), shape=(40, 40, 40))
    # Cell starting at +15 mm is 7.5 sigma from the spot.
    ix = 15 - (-20)
    assert vol[:, :, ix].sum() == 0.0
    assert vol.sum() == pytest.approx(1.0, rel=1e-3)


def test_grid_crops_at_512_cells() -> None:
    centers = np.array([[0.0, 0.0, 0.0], [2000.0, 0.0, 0.0]])
    sigmas = np.ones((2, 3))
    grid = dose_grid(centers, sigmas)
    assert grid.cropped
    assert grid.shape[0] == 512
    assert grid.shape[1] < 512


def _frame() -> PositionSigmaFrame:
    return PositionSigmaFrame(
        energy=np.array([70.0, 80.0]),
        ic1_x=np.array([1.0, 2.0]),
        ic1_y=np.array([3.0, 4.0]),
        ic2_x=np.array([10.0, 20.0]),
        ic2_y=np.array([30.0, 40.0]),
        ic1_sx=np.array([1.5, 1.6]),
        ic1_sy=np.array([1.7, 1.8]),
        ic2_sx=np.array([2.5, 2.6]),
        ic2_sy=np.array([2.7, 2.8]),
        weight=np.array([1.0, 1.0]),
        plan_x=np.array([0.0, 0.5]),
        plan_y=np.array([0.0, 0.5]),
    )


def test_measured_cloud_ic1_vs_ic2() -> None:
    frame = _frame()
    source = SessionSplatSource("s", iso=frame, chamber=frame, plan=None, n_raw=2)
    ic1 = measured_cloud(source, XY_IC1, "unused")
    ic2 = measured_cloud(source, XY_IC2, "unused")
    assert ic1 is not None and ic2 is not None
    np.testing.assert_allclose(ic1.x, [1.0, 2.0])
    np.testing.assert_allclose(ic2.x, [10.0, 20.0])
    np.testing.assert_allclose(ic1.sx, [1.5, 1.6])
    np.testing.assert_allclose(ic2.sx, [2.5, 2.6])


def test_measured_cloud_iso_ray_uses_chamber() -> None:
    iso = _frame()
    chamber = PositionSigmaFrame(
        energy=iso.energy,
        ic1_x=np.array([0.0, 0.0]),
        ic1_y=np.array([0.0, 0.0]),
        ic2_x=np.array([0.0, 10.0]),
        ic2_y=np.array([0.0, 0.0]),
        ic1_sx=np.ones(2),
        ic1_sy=np.ones(2),
        ic2_sx=np.ones(2),
        ic2_sy=np.ones(2),
        weight=np.ones(2),
    )
    source = SessionSplatSource("s", iso=iso, chamber=chamber, plan=None, n_raw=2)
    cloud = measured_cloud(source, XY_ISO_RAY, "unused")
    assert cloud is not None
    # Fallback iso z is IC1, so the ray lands on aligned IC1 (medians removed).
    assert cloud.x.size == 2
    assert np.isfinite(cloud.x).all()


def test_measured_cloud_iso_ray_falls_back_to_iso_frame() -> None:
    source = SessionSplatSource("s", iso=_frame(), chamber=None, plan=None, n_raw=2)
    cloud = measured_cloud(source, XY_ISO_RAY, "unused")
    assert cloud is not None
    assert cloud.x.size == 2
    assert np.isfinite(cloud.x).all()


def test_iso_ray_uses_other_ic_sigma_when_one_is_nan() -> None:
    iso = _frame()
    chamber = PositionSigmaFrame(
        energy=iso.energy,
        ic1_x=np.array([0.0, 0.0]),
        ic1_y=np.array([0.0, 0.0]),
        ic2_x=np.array([0.0, 10.0]),
        ic2_y=np.array([0.0, 0.0]),
        ic1_sx=np.array([3.0, 3.0]),
        ic1_sy=np.array([4.0, 4.0]),
        ic2_sx=np.array([np.nan, np.nan]),
        ic2_sy=np.array([np.nan, np.nan]),
        weight=np.ones(2),
    )
    source = SessionSplatSource("s", iso=iso, chamber=chamber, plan=None, n_raw=2)
    cloud = measured_cloud(source, XY_ISO_RAY, "unused")
    assert cloud is not None
    np.testing.assert_allclose(cloud.sx, [3.0, 3.0])
    np.testing.assert_allclose(cloud.sy, [4.0, 4.0])


def test_measured_cloud_plan_mode() -> None:
    plan = SplatCloud(
        x=np.array([5.0]),
        y=np.array([6.0]),
        sx=np.array([3.0]),
        sy=np.array([3.0]),
        energy=np.array([70.0]),
        weight=np.array([2.0]),
    )
    source = SessionSplatSource("s", iso=_frame(), chamber=None, plan=plan, n_raw=2)
    cloud = measured_cloud(source, XY_PLAN, "unused")
    assert cloud is plan


def test_difference_scale_cannot_stay_sequential() -> None:
    from scan_kit.views.dose_volume_catalog import active_scale
    from scan_kit.views.dose_volume_fill import colormap_samples

    assert active_scale(False, "coolwarm") == "viridis"
    assert active_scale(True, "viridis") == "coolwarm"
    assert active_scale(True, "seismic") == "seismic"
    assert active_scale(False, "magma") == "magma"
    rgb = colormap_samples("turbo", 3)
    assert rgb.shape == (3, 3)
    assert float(rgb[0, 2]) > float(rgb[-1, 2])


def test_residual_agreement_zero_is_transparent_or_white() -> None:
    assert SplatConfig().agreement == AGREE_TRANSPARENT
    clear = residual_signed_rgba(np.array([0.0, 1.0, -1.0]), scale=1.0, zero="transparent")
    np.testing.assert_allclose(clear[0, 3], 0.0, atol=1e-12)
    np.testing.assert_allclose(clear[1, :3], HOT_RGB)
    np.testing.assert_allclose(clear[1, 3], 1.0)
    np.testing.assert_allclose(clear[2, :3], COLD_RGB)
    np.testing.assert_allclose(clear[2, 3], 1.0)
    white = residual_signed_rgba(np.array([0.0]), scale=1.0, zero="white")
    np.testing.assert_allclose(white[0, :3], 1.0)
    np.testing.assert_allclose(white[0, 3], 1.0)
    # 2% on a ±10% scale is a light tint, not a saturated shell.
    assert residual_error_alpha(0.02, 0.10) < 0.25
    assert residual_error_alpha(0.10, 0.10) == 1.0
    assert residual_error_alpha(0.0, 0.10) == 0.0


def test_delivered_mu_falls_back_when_dose_is_missing() -> None:
    assert SplatConfig().error_mode == ERROR_ABSOLUTE
    np.testing.assert_allclose(
        dose_field_weights(np.array([0.4, np.nan]), np.array([1.0, 2.0])),
        [0.4, 2.0],
    )
    assert measured_dose_columns(
        ["ic1_total_dose_spot", "ic1_total_dose_spot_raw", "ic2_total_dose_spot_raw"],
    ) == {"ic1": "ic1_total_dose_spot"}


def test_axis_guides_point_depth_toward_high_energy() -> None:
    extent = np.array([[-10.0, -20.0, -330.0], [10.0, 20.0, -40.0]])
    _starts, ends, labels = axis_guide_points(extent, depth_sign=-1)
    assert ends[2, 2] < -330.0
    assert labels[2, 2] < ends[2, 2]
    assert ends[0, 0] > 10.0
    assert labels[0, 0] > ends[0, 0]
    _s2, e2, l2 = axis_guide_points(extent, depth_sign=1)
    assert e2[2, 2] > -40.0
    assert l2[2, 2] > e2[2, 2]


def test_gantry_90_swaps_y_and_energy_axis() -> None:
    pts = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(apply_gantry(pts, 0.0), pts, atol=1e-9)
    np.testing.assert_allclose(apply_gantry(pts, 90.0), [[1.0, -3.0, 2.0]], atol=1e-6)


def test_protons_from_ideal_air_ic() -> None:
    s70 = float(np.asarray(air_mass_stopping_mev_cm2_g(70.0)))
    s230 = float(np.asarray(air_mass_stopping_mev_cm2_g(230.0)))
    assert 8.5 < s70 < 10.0
    assert 4.0 < s230 < 4.6
    n10 = protons_from_charge_c(1e-8, 150.0, 10.0)
    n20 = protons_from_charge_c(1e-8, 150.0, 20.0)
    np.testing.assert_allclose(n10, 2.0 * n20, rtol=1e-6)
    assert protons_from_charge_c(1e-8, 230.0, 10.0) > protons_from_charge_c(
        1e-8, 70.0, 10.0,
    )
    k_mu = 2.0e-8
    np.testing.assert_allclose(
        protons_from_mu(1.0, 150.0, 10.0, k_mu),
        protons_from_charge_c(k_mu, 150.0, 10.0),
    )
    cloud = SplatCloud(
        x=np.array([0.0]), y=np.array([0.0]),
        sx=np.array([1.0]), sy=np.array([1.0]),
        energy=np.array([150.0]), weight=np.array([0.5]),
        k_mu=k_mu,
    )
    np.testing.assert_allclose(color_scalar(cloud, SplatConfig(weight_mode=WEIGHT_MU)), [0.5])
    ten = color_scalar(cloud, SplatConfig(weight_mode=WEIGHT_PROTONS, ic_gap_mm=10.0))
    five = color_scalar(cloud, SplatConfig(weight_mode=WEIGHT_PROTONS, ic_gap_mm=5.0))
    np.testing.assert_allclose(five, 2.0 * ten)
    assert parse_kmu_c_per_mu(
        '<MapToMap><devices><ion_chamber><device name="IC_1_X"/>'
        '<gain_conversion in_units="MU" K_MU="2.5e-08"/></ion_chamber></devices></MapToMap>'
    ) == 2.5e-8
    assert parse_kmu_c_per_mu("<nope/>") is None


def test_energy_rgb_yellow_is_high() -> None:
    from matplotlib import cm

    rgb = energy_rgb(np.array([70.0, 230.0]), vmin=70.0, vmax=230.0)
    np.testing.assert_allclose(rgb[0], cm.viridis(0.0)[:3], atol=1e-5)
    np.testing.assert_allclose(rgb[1], cm.viridis(1.0)[:3], atol=1e-5)
    assert rgb[1, 0] > rgb[0, 0]
    assert rgb[0, 2] > rgb[1, 2]


def test_dose_volume_module_has_run() -> None:
    from scan_kit.views import dose_volume

    assert callable(dose_volume.run)


def test_ordered_float_matches_the_reduce_sentinels() -> None:
    from scan_kit.views.dose_volume_raycast import (
        ORDERED_NEG_INF,
        ORDERED_POS_INF,
        float_from_ordered,
        ordered_from_float,
    )

    assert ordered_from_float(np.inf) == ORDERED_POS_INF
    assert ordered_from_float(-np.inf) == ORDERED_NEG_INF
    values = [-np.inf, -100.0, -1.0, -0.0, 0.0, 1.0, 100.0, np.inf]
    ordered = [ordered_from_float(v) for v in values]
    assert ordered == sorted(ordered)
    for value in values:
        got = float_from_ordered(ordered_from_float(value))
        assert got == value or (value == 0.0 and got == 0.0)


def test_auto_color_range_dose_keeps_zero() -> None:
    from scan_kit.views.dose_volume_raycast import (
        auto_color_range,
        manual_color_limits,
        suggest_abs_window,
    )

    assert auto_color_range(False, -5.0, 2.0) == (0.0, 2.0)
    assert auto_color_range(False, -2.0, -1.0) is None
    assert auto_color_range(False, 0.0, 0.0) is None
    assert auto_color_range(False, np.nan, 1.0) is None
    assert auto_color_range(True, -0.2, 0.5) == (-0.2, 0.5)
    flat = auto_color_range(True, 0.25, 0.25)
    assert flat is not None and flat[0] < 0.25 < flat[1]
    lo, hi = manual_color_limits(
        difference=False, transparent=False, integral=True,
        gain=0.5, typical=0.1, ray_scale=2.0, error_scale=0.2, absolute=True,
    )
    assert (lo, hi) == pytest.approx((0.0, 4.0))
    dlo, dhi = manual_color_limits(
        difference=True, transparent=False, integral=True,
        gain=0.5, typical=0.1, ray_scale=2.0, error_scale=0.2, absolute=True,
    )
    assert (dlo, dhi) == pytest.approx((-0.2, 0.2))
    tlo, thi = manual_color_limits(
        difference=True, transparent=True, integral=False,
        gain=0.5, typical=0.1, ray_scale=2.0, error_scale=0.2, absolute=True,
    )
    assert (tlo, thi) == pytest.approx((-0.2, 0.2))
    # A fixed 0.2 MU window is far above a small fluence and washes out to white.
    assert suggest_abs_window(0.0064) == pytest.approx(0.005)
    assert suggest_abs_window(0.0064) < 0.2


def test_color_axis_ticks_cover_the_span_and_zero() -> None:
    from scan_kit.views.dose_volume_window import color_axis_ticks

    majors, minors, labels, _offset = color_axis_ticks(-0.2, 0.5)
    assert majors[0] >= -0.2 - 1e-6
    assert majors[-1] <= 0.5 + 1e-6
    assert any(abs(float(v)) < 1e-9 for v in majors)
    assert len(labels) == len(majors)
    assert len(minors) >= 4
    assert all(lo < float(v) < hi for v in minors for lo, hi in [(-0.2, 0.5)])
    small, _, small_labels, offset = color_axis_ticks(0.0, 0.00224)
    assert small[0] >= -1e-9 and small[-1] <= 0.00224 + 1e-9
    assert offset or all(len(label) <= 8 for label in small_labels)


def test_color_axis_paints_the_scale(qapp) -> None:
    from PySide6.QtGui import QImage

    from scan_kit.views.dose_volume_window import _AXIS_PAD_X, _ColorAxis

    axis = _ColorAxis()
    axis.resize(axis.sizeHint().width(), 360)
    axis.set_scale("viridis", 0.0, 1.0, "MU / mm²")
    assert axis.width() == axis.sizeHint().width()
    axis.set_scale("coolwarm", -0.00021, 0.00224, "meas − plan (MU)")
    assert axis.width() == axis.sizeHint().width()
    image = QImage(axis.size(), QImage.Format.Format_RGB32)
    image.fill(0)
    axis.render(image)
    top = image.pixelColor(_AXIS_PAD_X + 4, 56)
    bottom = image.pixelColor(_AXIS_PAD_X + 4, 300)
    assert (top.red(), top.green(), top.blue()) != (bottom.red(), bottom.green(), bottom.blue())
    axis.deleteLater()
    qapp.processEvents()


def test_dose_volume_window_opens_with_absolute_scale(qapp, tmp_path) -> None:
    from PySide6.QtWidgets import QLabel

    from scan_kit.views.dose_volume_catalog import DEFAULT_ERROR_MU, RAY_TRANSPARENT
    from scan_kit.views.dose_volume_window import DoseVolumeWindow, _log_slider_pos

    window = DoseVolumeWindow([], str(tmp_path))
    try:
        assert window._weight_combo.currentData() == WEIGHT_MU
        assert window._error_combo.currentData() == ERROR_ABSOLUTE
        assert window._error_scale_spin.value() == pytest.approx(DEFAULT_ERROR_MU)
        assert window._error_scale_spin.suffix() == " MU/mm²"
        assert window._scale_mode_row.isHidden()
        assert window._grain_combo.parentWidget().findChild(QLabel).text() == "Grain"
        assert window._cap_spin.parentWidget().findChild(QLabel).text() == "Spot cap"
        gutter = window._color_axis.parentWidget()
        assert gutter.layout().indexOf(window._color_axis) == 0
        assert gutter.layout().indexOf(window._side_scroll) == 1
        assert window._splitter.widget(1) is gutter
        assert window._color_axis.minimumWidth() == window._color_axis.maximumWidth()
        assert window._auto_check.isChecked()
        assert not window._gain_box.isHidden()
        assert not window._gain_slider.isEnabled()
        window._updating = True
        window._set_combo(window._show_combo, "difference")
        window._updating = False
        window._sync_color_controls()
        assert not window._gain_box.isHidden()
        assert not window._scale_mode_row.isHidden()
        assert not window._error_scale_spin.isHidden()
        assert not window._gain_slider.isEnabled()
        window._auto_check.setChecked(False)
        assert not window._scale_mode_row.isHidden()
        assert window._gain_slider.isEnabled()
        lo = window._error_scale_spin.minimum()
        hi = window._error_scale_spin.maximum()
        assert window._gain_slider.value() == _log_slider_pos(
            window._error_scale_spin.value(), lo, hi,
        )
        window._error_scale_spin.setValue(0.5)
        assert window._gain_slider.value() == _log_slider_pos(0.5, lo, hi)
        window._gain_slider.setValue(_log_slider_pos(0.05, lo, hi))
        assert window._error_scale_spin.value() == pytest.approx(0.05, rel=0.02)
        window._updating = True
        window._set_combo(window._ray_combo, RAY_TRANSPARENT)
        window._auto_check.setChecked(True)
        window._updating = False
        window._sync_color_controls()
        assert window._gain_slider.isEnabled()
        assert not window._scale_mode_row.isHidden()
        assert window._error_scale_spin.isEnabled()
        assert window._window_label.text() == "Opacity"
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_shift_fov_drag_ignores_leftover_zoom_tuple() -> None:
    from vispy.util import keys

    from scan_kit.views.dose_volume_vispy import make_dose_camera

    camera = make_dose_camera()
    camera._event_value = (camera.scale_factor, camera.distance)
    camera._gesture = (frozenset({2}), frozenset())

    class _Mouse:
        modifiers = (keys.SHIFT,)

        class press_event:
            pos = np.array([0.0, 0.0])

        pos = np.array([0.0, 10.0])

    class _Event:
        handled = False
        type = "mouse_move"
        buttons = {2}
        press_event = object()
        mouse_event = _Mouse()

    camera.viewbox_mouse_event(_Event())
    assert camera.fov == pytest.approx(45.0 - 10.0 / 5.0)


def test_offscreen_dose_volume_render(qapp) -> None:
    from scan_kit.views.dose_volume_data import LinearEnergyAxis, SplatBatch
    from scan_kit.views.dose_volume_vispy import DoseScene
    from scan_kit.views.vispy_plot import make_scene_canvas

    try:
        canvas = make_scene_canvas(size=(160, 120), show=False, gl="gl+")
    except Exception:
        try:
            canvas = make_scene_canvas(size=(160, 120), show=False)
        except Exception as exc:
            pytest.skip(f"visPy canvas unavailable: {exc}")
    try:
        batch = SplatBatch(
            center=np.zeros((1, 3), dtype=np.float32),
            sigma=np.full((1, 3), 4.0, dtype=np.float32),
            weight=np.array([8.0], dtype=np.float32),
            rgb=np.ones((1, 3), dtype=np.float32),
            energy_mev=np.zeros(1, dtype=np.float32),
        )
        scene = DoseScene(canvas)
        scene.render(
            batch, None, LinearEnergyAxis(mm_per_mev=1.0),
            gain=1.0, gantry_deg=0.0, smear=4.0,
        )
        from vispy import gloo

        width, height = (int(v) for v in canvas.size)
        fbo = gloo.FrameBuffer(
            color=gloo.RenderBuffer((height, width)),
            depth=gloo.RenderBuffer((height, width)),
        )
        canvas.push_fbo(fbo, (0, 0), (width, height))
        try:
            canvas.on_draw(None)
            img = np.asarray(fbo.read())
        finally:
            canvas.pop_fbo()
        broken = scene._broken
        auto_hi = scene.auto_hi
    except Exception as exc:
        pytest.skip(f"visPy cannot ray march offscreen: {exc}")
    assert not broken
    assert auto_hi is not None and auto_hi > 0.0
    if img.ndim != 3 or img.size == 0:
        pytest.skip("visPy render returned an empty image")
    assert img.shape[0] >= 80
    assert img[..., :3].max() > 0
