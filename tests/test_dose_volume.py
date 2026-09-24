"""Tests for the dose-volume grid, loaders, and ray-march smoke."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from scan_kit.common.ic_trajectory import IC1_Z_MM, IC2_Z_MM, IC_SEP_MM
from scan_kit.views.dose_volume_catalog import (
    ERROR_ABSOLUTE,
    SplatConfig,
    WEIGHT_DOSE,
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
    concat_clouds,
    iso_xy_from_ic_ray,
    measured_cloud,
    measured_dose_columns,
    parse_kmu_c_per_mu,
    protons_from_charge_c,
    protons_from_mu,
    range_axis_for_medium,
)
from scan_kit.views.dose_volume_fill import (
    DoseGrid,
    dose_field_weights,
    dose_grid,
    deposit_gaussians,
    normal_mass,
)
from scan_kit.views.dose_volume_vispy import (
    apply_gantry, axis_ticks, box_axes, label_anchors, label_direction, outward_index,
    readable_angle, spaced_labels,
)


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
    batch = cloud_to_batch(cloud, WetStub(), smear_axis_units=2.0)
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
    batch = cloud_to_batch(cloud, LinearEnergyAxis(1.0), 0.5)
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
    np.testing.assert_allclose(z[1], -77.2, atol=0.5)
    copper = range_axis_for_medium("copper")
    z_cu = copper.z_scene_mm(np.array([100.0]))
    assert 1.0 / 8.96 < z_cu[0] / z[1] < 1.6 / 8.96
    # Straggling alone is ~1.1 % of range; 1 % energy spread adds ~1.7 %.
    bare = water.sigma_z_scene_mm(np.array([100.0]), 0.0)[0]
    spread = water.sigma_z_scene_mm(np.array([100.0]), 1.0)[0]
    assert 0.009 < bare / abs(z[1]) < 0.014
    assert spread > 1.5 * bare
    assert water.smear_label == "%"
    assert water.depth_sign == -1
    assert "water" in water.axis_label.lower()
    assert "copper" in copper.axis_label.lower()
    pmma = range_axis_for_medium("pmma")
    aluminum = range_axis_for_medium("aluminum")
    assert "PMMA" in pmma.axis_label and "aluminum" in aluminum.axis_label
    # Plastics stop in about the same mass as water; aluminum is much shorter in mm.
    assert abs(pmma.z_scene_mm(np.array([100.0]))[0]) < abs(z[1])
    assert abs(aluminum.z_scene_mm(np.array([100.0]))[0]) < abs(pmma.z_scene_mm(np.array([100.0]))[0])


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


def test_coarse_voxels_still_conserve_mu() -> None:
    from scan_kit.views.dose_volume_vispy import box_edge_segments

    grid = DoseGrid(np.array([-8.0, -8.0, -8.0]), (8, 8, 8), False, voxel=2.0)
    vol = deposit_gaussians(
        np.zeros((1, 3)), np.full((1, 3), 2.0), np.array([3.0]), np.zeros(1),
        _ZAtEnergy(2.0), grid,
    )
    assert vol.shape == (8, 8, 8)
    assert vol.sum() == pytest.approx(3.0, rel=1e-3)
    edges = box_edge_segments(grid.origin, grid.extent_mm)
    assert edges.shape == (24, 3)
    lengths = np.linalg.norm(edges[1::2] - edges[::2], axis=1)
    np.testing.assert_allclose(lengths, 16.0)


def test_background_is_the_scale_zero_color() -> None:
    from matplotlib import colormaps

    from scan_kit.views.dose_volume_fill import ink_rgb, zero_rgb

    np.testing.assert_allclose(zero_rgb("viridis", 0.0, 5.0), colormaps["viridis"](0.0)[:3])
    np.testing.assert_allclose(zero_rgb("PuOr_r", -2.0, 2.0), colormaps["PuOr_r"](0.5)[:3])
    np.testing.assert_allclose(zero_rgb("berlin", -1.0, 3.0), colormaps["berlin"](0.25)[:3])
    np.testing.assert_allclose(zero_rgb("viridis", 1.0, 5.0), colormaps["viridis"](0.0)[:3])
    assert sum(ink_rgb((1.0, 1.0, 1.0))) < 1.0
    assert sum(ink_rgb((0.0, 0.0, 0.0))) > 2.0


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


def test_plan_sigma_sources_and_iso_plane() -> None:
    from dataclasses import replace

    from scan_kit.views.dose_volume_catalog import (
        PLAN_SIGMA_INTERLOCK, PLAN_SIGMA_REFERENCE, SIGMA_PLANE_ISO,
    )
    from scan_kit.views.dose_volume_data import plan_cloud

    class _Geom:
        def mag_factor(self, device):
            return {"IC_1_X": 2.0, "IC_1_Y": 3.0, "IC_2_X": 1.5, "IC_2_Y": 1.5}[device]

    plan = SplatCloud(
        x=np.zeros(3), y=np.zeros(3), sx=np.full(3, 5.0), sy=np.full(3, 5.0),
        energy=np.array([70.0, 80.0, 75.0]), weight=np.ones(3),
    )
    source = SessionSplatSource("s", iso=_frame(), chamber=None, plan=plan, n_raw=2, geom=_Geom())
    iso = measured_cloud(source, XY_IC1, plane=SIGMA_PLANE_ISO)
    np.testing.assert_allclose(iso.sx, [3.0, 3.2])
    np.testing.assert_allclose(iso.sy, [5.1, 5.4])

    config = SplatConfig()
    got = plan_cloud(source, config)
    np.testing.assert_allclose(got.sx, [1.5, 1.6, 1.55])
    np.testing.assert_allclose(got.sy, [1.7, 1.8, 1.75])
    got = plan_cloud(source, replace(config, sigma_plane=SIGMA_PLANE_ISO))
    np.testing.assert_allclose(got.sx, [3.0, 3.2, 3.1])

    wide = replace(_frame(), ic1_sx=np.array([4.0, 6.0]), ic1_sy=np.array([4.0, 6.0]))
    ref = SessionSplatSource("r", iso=wide, chamber=None, plan=None, n_raw=2)
    got = plan_cloud(source, replace(config, plan_sigma=PLAN_SIGMA_REFERENCE), ref)
    np.testing.assert_allclose(got.sx, [4.0, 6.0, 5.0])

    interlock = replace(config, plan_sigma=PLAN_SIGMA_INTERLOCK)
    np.testing.assert_allclose(plan_cloud(source, interlock).sx, 5.0)
    got = plan_cloud(source, replace(interlock, sigma_plane=SIGMA_PLANE_ISO))
    np.testing.assert_allclose(got.sx, 10.0)
    np.testing.assert_allclose(got.sy, 15.0)


def test_difference_scale_cannot_stay_sequential() -> None:
    from scan_kit.views.dose_volume_catalog import active_scale
    from scan_kit.views.dose_volume_fill import colormap_samples

    assert active_scale(False, "coolwarm") == "turbo"
    assert active_scale(True, "viridis") == "managua_r"
    assert active_scale(True, "berlin") == "berlin"
    assert active_scale(False, "magma") == "magma"
    rgb = colormap_samples("turbo", 3)
    assert rgb.shape == (3, 3)
    assert float(rgb[0, 2]) > float(rgb[-1, 2])


def test_delivered_mu_falls_back_when_dose_is_missing() -> None:
    assert SplatConfig().error_mode == ERROR_ABSOLUTE
    np.testing.assert_allclose(
        dose_field_weights(np.array([0.4, np.nan]), np.array([1.0, 2.0])),
        [0.4, 2.0],
    )
    assert measured_dose_columns(
        ["ic1_total_dose_spot", "ic1_total_dose_spot_raw", "ic2_total_dose_spot_raw"],
    ) == {"ic1": "ic1_total_dose_spot"}


def test_box_axes_tick_every_cm_on_the_box_edges() -> None:
    vals, labeled = axis_ticks(-25.0, 31.0)
    np.testing.assert_allclose(vals, [-25.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 31.0])
    assert labeled[0] and labeled[-1]
    vals, labeled = axis_ticks(-330.0, 0.0)
    assert vals.size == 34 and labeled.sum() <= 8 and labeled[[0, -1]].all()
    # The end always reads; a round number crowding it gives way.
    vals, labeled = axis_ticks(-320.0, 0.0)
    assert labeled[0] and not labeled[vals == -300.0].any()
    assert spaced_labels([[0, 0], [10, 0], [20, 0]], 8.0) == [0, 1, 2]
    assert spaced_labels([[0, 0], [5, 0], [10, 0]], 8.0) == [0, 2]
    assert spaced_labels([[0, 0], [2, 0], [4, 0]], 8.0) == [0, 2]
    # A tick jammed against the end drops only itself, not the labels that still fit.
    assert spaced_labels([[0, 0], [20, 0], [40, 0], [42, 0]], 8.0) == [0, 1, 3]

    origin, extent = np.array([-15.0, -20.0, -330.0]), np.array([30.0, 40.0, 332.0])
    guides = box_axes(origin, extent, ("X", "Y", "Depth"), depth_sign=-1)
    starts = guides.ticks[0::2]
    ends = guides.ticks[1::2]
    assert starts.shape[0] == 5 + 5 + 35
    # X ticks sit on the surface edge (y = lo, z = hi) and point out of the box along −y.
    x = starts[:5]
    np.testing.assert_allclose(x[:, 1:], [[-20.0, 2.0]] * 5)
    assert (ends[:5, 1] < -20.0).all()
    # Depth numbers read positive and hang off tick tips outside the box.
    x_ax, _y_ax, z_ax = guides.axes
    assert z_ax.numbers[0] == "330" and z_ax.numbers[-2:] == ["0", "-2"]
    assert (z_ax.tips[:, 0] < -15.0).all() and np.allclose(z_ax.tips[:, 1], -20.0)
    np.testing.assert_allclose(x_ax.out, [0.0, -1.0, 0.0])
    assert [a.title for a in guides.axes] == ["X", "Y", "Depth"]
    assert label_anchors((-1.0, 0.1)) == ("right", "center")
    assert label_anchors((0.0, 1.0)) == ("center", "top")
    # A nearly-parallel outward still hangs labels square off the edge, on its side.
    np.testing.assert_allclose(label_direction((1.0, 0.0), (-0.9, 0.1)), [0.0, 1.0])
    np.testing.assert_allclose(label_direction((0.0, 0.0), (0.0, -2.0)), [0.0, -1.0])
    # A side view: the outward that runs along the edge loses to the one that leaves it.
    assert outward_index((10.0, 0.0), [(1.0, 0.0), (0.0, -1.0)], (0.0, -1.0)) == 1
    assert outward_index((10.0, 0.0), [(0.0, 1.0), (1.0, 0.0)], (0.0, 1.0)) == 0
    assert readable_angle((1.0, 0.0)) == pytest.approx(0.0)
    assert readable_angle((-1.0, 0.0)) == pytest.approx(0.0)
    assert readable_angle((0.0, 1.0)) == pytest.approx(90.0)
    assert readable_angle((0.0, -1.0)) == pytest.approx(-90.0)


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
    from scan_kit.views.dose_volume_data import SplatBatch
    from scan_kit.views.dose_volume_vispy import deposit_amounts

    # Delivered MU wins over the plan weight; dose starts from the same protons as stops.
    batch = SplatBatch(
        center=np.zeros((2, 3)), sigma=np.ones((2, 3)), weight=np.array([1.0, 1.0]),
        energy_mev=np.array([150.0, 150.0]), dose_mu=np.array([0.5, np.nan]), k_mu=k_mu,
    )
    np.testing.assert_allclose(deposit_amounts(batch, WEIGHT_MU, 10.0), [0.5, 1.0])
    protons = deposit_amounts(batch, WEIGHT_PROTONS, 10.0)
    np.testing.assert_allclose(deposit_amounts(batch, WEIGHT_DOSE, 10.0), protons)
    np.testing.assert_allclose(deposit_amounts(batch, WEIGHT_PROTONS, 5.0), 2.0 * protons, rtol=1e-6)
    assert parse_kmu_c_per_mu(
        '<MapToMap><devices><ion_chamber><device name="IC_1_X"/>'
        '<gain_conversion in_units="MU" K_MU="2.5e-08"/></ion_chamber></devices></MapToMap>'
    ) == 2.5e-8
    assert parse_kmu_c_per_mu("<nope/>") is None
    chambers = "".join(
        f'<ion_chamber><device name="{name}"/><gain_conversion in_units="MU" K_MU="{k}"/></ion_chamber>'
        for name, k in (("IC_1_X", "2.5e-08"), ("IC_1_HCC", "2.6e-08"), ("IC_2_HCC", "2.7e-08"))
    )
    xml = f"<MapToMap><devices>{chambers}</devices></MapToMap>"
    assert parse_kmu_c_per_mu(xml) == 2.6e-8
    assert parse_kmu_c_per_mu(xml, "IC_2") == 2.7e-8


def test_timeslice_mu_from_scan_dose_counter() -> None:
    from scan_kit.views.dose_volume_data import nonnegative_layer_mu, timeslice_charge_nc

    got = timeslice_charge_nc([np.nan, 0.0, 1.0, np.nan, 3.0, 2.9, 6.0])
    np.testing.assert_allclose(got, [0.0, 0.0, 1.0, 0.0, 2.0, -0.1, 3.1])
    assert got.sum() == pytest.approx(6.0)
    energy = np.array([70.0, 70.0, 70.0, 80.0, 80.0])
    mu = nonnegative_layer_mu(np.array([1.0, -0.2, 1.2, 0.5, np.nan]), energy)
    assert (mu >= 0.0).all()
    assert mu[:3].sum() == pytest.approx(2.0) and mu[3:].sum() == pytest.approx(0.5)

    from scan_kit.views.dose_volume_data import fill_from_spot

    # Spot 1 ramps up unfitted; spot 2 never fits and takes the next sample; layer 80 stands alone.
    x = np.array([np.nan, 1.0, 3.0, np.nan, np.nan, 7.0, np.nan, 9.0])
    spot = np.array([1, 1, 1, 2, 2, 3, 4, 4])
    energy = np.array([70.0] * 6 + [80.0] * 2)
    np.testing.assert_allclose(fill_from_spot(x, spot, energy), [2.0, 1.0, 3.0, 7.0, 7.0, 7.0, 9.0, 9.0])


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
    assert auto_color_range(True, -0.2, 0.5) == (-0.5, 0.5)
    assert auto_color_range(True, -0.7, 0.1) == (-0.7, 0.7)
    flat = auto_color_range(True, 0.0, 0.0)
    assert flat is not None and flat[0] < 0.0 < flat[1]
    assert auto_color_range(True, -1e-8, 2e-8, 0.01) == (-0.01, 0.01)
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
    assert majors[0] == -0.2 and majors[-1] == 0.5
    ends, _, end_labels, _ = color_axis_ticks(0.013, 0.587)
    assert (ends[0], ends[-1]) == (0.013, 0.587)
    assert end_labels[0] == "0.013" and end_labels[-1] == "0.587"
    assert np.all(np.diff(ends) > 0)
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
        assert window._weight_combo.current_key() == WEIGHT_DOSE
        assert window._error_combo.currentData() == ERROR_ABSOLUTE
        assert window._error_scale_spin.value() == pytest.approx(DEFAULT_ERROR_MU)
        assert window._error_scale_spin.suffix() == " Gy·mm"
        assert window._smear_spin.suffix() == " %"
        assert window._gap_spin.isEnabled()
        assert window._gamma_group.isHidden()
        def group_of(widget):
            return widget.parentWidget().parentWidget().title()

        assert [group_of(w) for w in (window._grain_combo, window._gap_spin, window._cap_spin)] == [
            "Beam", "Compare", "View",
        ]
        assert [group_of(w) for w in (window._medium_combo, window._margin_spin, window._wet_spin)] == ["Phantom"] * 3
        assert [group_of(w) for w in (window._weight_combo, window._voxel_spin)] == ["Compare", "View"]
        assert [group_of(w) for w in (window._plane_combo, window._plan_sigma_combo)] == ["Beam"] * 2
        assert [group_of(w) for w in (window._grid_label, window._spots_label)] == ["View"] * 2
        assert window._scatter_check.parentWidget().title() == "Beam"
        assert not window._phantom_box_check.isChecked() and window._field_box_check.isChecked()
        assert window._phantom_box_check.parentWidget() is window._phantom_note.parentWidget()
        assert window._field_box_check.parentWidget() is window._field_size.parentWidget()
        # Session radios sit above Beam, show only for 2+, and the pick survives a rebuild.
        assert window._session_group.isHidden()
        window._update_session_list(["s1", "s2"])
        column = window._session_group.parentWidget().layout()
        beam = window._grain_combo.parentWidget().parentWidget()
        assert column.indexOf(window._session_group) < column.indexOf(beam)
        assert not window._session_group.isHidden()
        radios = window._session_buttons.buttons()
        assert radios[0].isChecked() and window._active_session == "s1"
        radios[1].click()
        window._update_session_list(["s1", "s2"])
        assert window._active_session == "s2" and radios[1].isChecked()
        window._update_session_list(["s1"])
        assert window._active_session == "s1" and window._session_group.isHidden()
        assert window._phantom_spin.text() == "Auto"
        assert window._margin_spin.value() == 5.0 and window._margin_spin.isEnabled()
        window._phantom_spin.setValue(200.0)
        assert not window._margin_spin.isEnabled()
        window._phantom_spin.setValue(0.0)
        assert window._wet_spin.value() == 0.0
        assert window._scale_mode_row.isHidden()
        assert window._grain_combo.parentWidget().findChild(QLabel).text() == "Data"
        assert window._cap_spin.parentWidget().findChild(QLabel).text() == "Spot cap"
        # The splitter handle sits between the axis and the controls.
        view = window._color_axis.parentWidget()
        assert view.layout().indexOf(window._plot_host) == 0
        assert view.layout().indexOf(window._color_axis) == 1
        assert window._splitter.widget(0) is view
        assert window._splitter.widget(1) is window._side_scroll
        assert window._voxel_spin.value() == 1.0
        assert window._interp.current_key() == "linear"
        assert window._color_axis.minimumWidth() == window._color_axis.maximumWidth()
        assert window._auto_check.isChecked()
        assert not window._gain_box.isHidden()
        assert not window._gain_slider.isEnabled()
        window._updating = True
        window._set_combo(window._show_combo, "difference")
        window._updating = False
        window._sync_color_controls()
        # No plan loaded, so the volume is still dose and the axis must say so.
        window._update_legend()
        assert window._color_axis._title == "Gy·mm"
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
        window._updating = True
        window._set_combo(window._show_combo, "gamma")
        window._updating = False
        window._sync_color_controls()
        assert not window._gamma_group.isHidden()
        assert window._gain_box.isHidden()
        assert not window._ray_combo.isEnabled()
        config = window._read_config()
        assert config.gamma and config.overlay_plan
        assert (config.gamma_dose_pct, config.gamma_dta_mm, config.gamma_cutoff_pct) == (3.0, 2.0, 10.0)
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


def test_plan_vs_plan_reports_zero_error(qapp) -> None:
    from scan_kit.views.dose_volume_data import SplatBatch
    from scan_kit.views.dose_volume_raycast import read_texture
    from scan_kit.views.dose_volume_vispy import DoseScene
    from scan_kit.views.vispy_plot import make_scene_canvas

    try:
        canvas = make_scene_canvas(size=(64, 64), show=False, gl="gl+")
        canvas.set_current()
    except Exception as exc:
        pytest.skip(f"visPy canvas unavailable: {exc}")
    rng = np.random.default_rng(3)
    n = 40
    energy = rng.choice([80.0, 120.0, 180.0], n).astype(np.float32)
    plan = SplatBatch(
        center=np.column_stack([rng.uniform(-20, 20, (n, 2)), np.zeros(n)]).astype(np.float32),
        sigma=np.full((n, 3), 3.5, dtype=np.float32),
        weight=rng.uniform(0.5, 2.0, n).astype(np.float32),
        energy_mev=energy,
        dose_mu=rng.uniform(0.5, 2.0, n).astype(np.float32),
        k_mu=2.6e-8,
    )
    scene = DoseScene(canvas)
    for mode in (WEIGHT_MU, WEIGHT_PROTONS, WEIGHT_DOSE):
        scene.render(
            plan, plan, range_axis_for_medium("water"), gain=1.0, gantry_deg=0.0,
            weight_mode=mode, voxel_mm=2.0, gamma=True,
        )
        shape = tuple(scene._meas_tex.shape[:3])
        meas = read_texture(canvas, scene._meas_tex, shape)
        ref = read_texture(canvas, scene._plan_tex, shape)
        # The GPU tile lists fill in any order, so sums differ only by float rounding.
        assert meas.max() > 0.0 and np.abs(meas - ref).max() <= 1e-6 * meas.max(), mode
        passed, total = scene.gamma_pass
        assert total > 0 and passed == total, mode
    # Last mode was Gy with scatter; without it the same spots stay narrower and peak higher.
    scene.render(
        plan, plan, range_axis_for_medium("water"), gain=1.0, gantry_deg=0.0,
        weight_mode=WEIGHT_DOSE, voxel_mm=2.0, gamma=True, scatter=False,
    )
    passed, total = scene.gamma_pass
    shape = tuple(scene._meas_tex.shape[:3])
    assert passed == total and read_texture(canvas, scene._meas_tex, shape).max() > meas.max()
    scene.render(plan, plan, range_axis_for_medium("water"), gain=1.0, difference=True)
    scene._consume_auto_span((-1e-7, 1e-7))
    assert scene.auto_hi == pytest.approx(0.01 * scene.ray_peak)


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
        from OpenGL.GL import GL_CURRENT_PROGRAM, glGetIntegerv

        canvas.push_fbo(fbo, (0, 0), (width, height))
        try:
            canvas.on_draw(None)
        except Exception as exc:
            canvas.pop_fbo()
            pytest.skip(f"visPy cannot ray march offscreen: {exc}")
    except Exception as exc:
        pytest.skip(f"visPy cannot ray march offscreen: {exc}")
    try:
        first = np.asarray(fbo.read())
        env = canvas.context.shared.parser.env.get("current_program")
        real = int(np.asarray(glGetIntegerv(GL_CURRENT_PROGRAM)).reshape(-1)[0])
        # A uniform queued after the march must land on the box program.
        scene.set_gain(0.5)
        canvas.on_draw(None)
        img = np.asarray(fbo.read())
    finally:
        canvas.pop_fbo()
    broken = scene._broken
    auto_hi = scene.auto_hi
    # The box and its ticks; each must depth test against the march, not inherit its "always".
    lines = [node for node in scene._late if hasattr(node, "_line_visual")]
    assert len(lines) == 3
    # Labels land on the canvas, not collapsed to a point by the perspective divide.
    numbers = next(n for n, _title, _ax in scene._labels if n is not None)
    assert np.ptp(np.asarray(numbers.pos)[:, :2], axis=0).max() > 5.0
    assert all(line._line_visual._vshare.gl_state.get("depth_func") == "lequal" for line in lines)
    assert env == real
    assert not broken
    assert first[..., :3].max() > 0
    assert auto_hi is not None and auto_hi > 0.0
    if img.ndim != 3 or img.size == 0:
        pytest.skip("visPy render returned an empty image")
    assert img.shape[0] >= 80
    assert img[..., :3].max() > 0


def test_water_range_matches_pstar() -> None:
    from scan_kit.views.dose_volume_physics import WATER, csda_range_mm

    # NIST PSTAR CSDA ranges in water, g/cm² (= cm).
    for energy, pstar_cm in ((70.0, 4.080), (100.0, 7.718), (150.0, 15.77), (200.0, 25.96)):
        assert float(csda_range_mm(WATER, energy)) == pytest.approx(pstar_cm * 10.0, rel=0.005)


def test_copper_stops_in_less_mass_than_density_scaling_assumes() -> None:
    from scan_kit.views.dose_volume_physics import COPPER, WATER, csda_range_mm

    # Higher I and lower Z/A: copper needs ~1.5x the areal mass of water.
    ratio = float(csda_range_mm(COPPER, 150.0)) * COPPER.rho / float(csda_range_mm(WATER, 150.0))
    assert 1.4 < ratio < 1.6


def test_bragg_curve_conserves_energy_and_r80_is_range() -> None:
    from scan_kit.views.dose_volume_physics import (
        WATER, bragg_idd, csda_range_mm, local_energy_fraction,
    )

    e0 = 150.0
    r0 = float(csda_range_mm(WATER, e0))
    z = np.linspace(0.0, r0 + 30.0, 20001)
    idd = bragg_idd(WATER, e0, 1.0, z)
    total = float(np.sum(0.5 * (idd[1:] + idd[:-1]) * np.diff(z)))
    assert total / e0 == pytest.approx(local_energy_fraction(WATER, e0), rel=2e-3)
    peak = int(np.argmax(idd))
    distal = z[peak:][idd[peak:] <= 0.8 * idd[peak]][0]
    assert distal == pytest.approx(r0, abs=0.5)
    assert 0.2 < idd[0] / idd[peak] < 0.35


def test_scatter_at_end_of_range_near_preston_koehler() -> None:
    from scan_kit.views.dose_volume_physics import WATER, csda_range_mm, mcs_sigma_mm

    for energy in (100.0, 160.0, 230.0):
        r0 = float(csda_range_mm(WATER, energy))
        pk_mm = 0.0294 * (r0 / 10.0) ** 0.896 * 10.0
        assert float(mcs_sigma_mm(WATER, energy, r0)) == pytest.approx(pk_mm, rel=0.08)


def test_layer_kernel_conserves_mass_and_widens_with_depth() -> None:
    from scan_kit.views.dose_volume_physics import WATER, LayerDoseKernel, build_layer_tables

    kernel = LayerDoseKernel(build_layer_tables(WATER, [120.0], 1.0, nodes=512))
    center = np.array([[0.0, 0.0, 0.0]])
    sigma = np.array([[3.0, 3.0, 1.0]])
    c, s = kernel.span(center, sigma, [120.0])
    grid = dose_grid(c, s, voxel_mm=1.0)
    vol = deposit_gaussians(center, sigma, np.array([2.0]), np.array([120.0]), kernel, grid)
    assert float(vol.sum()) == pytest.approx(2.0, rel=1e-3)
    # Rows are z; the deepest dose is wider than at the surface.
    rows = vol.sum(axis=1)
    width = lambda row: float(np.sum(row > 0.5 * row.max()))  # noqa: E731
    live = [i for i in range(rows.shape[0]) if rows[i].max() > 0.2 * rows.max()]
    assert width(rows[live[0]]) > width(rows[live[-1]])


def test_finite_phantom_keeps_only_the_dose_inside() -> None:
    from scan_kit.views.dose_volume_physics import WATER, LayerDoseKernel, build_layer_tables

    # 150 MeV ranges out at ~158 mm, so a 100 mm phantom holds the plateau only.
    kernel = LayerDoseKernel(build_layer_tables(WATER, [150.0], 1.0, nodes=512))
    center, sigma, energy = np.zeros((1, 3)), np.full((1, 3), 3.0), np.array([150.0])
    c, s = kernel.span(center, sigma, energy)
    grid = dose_grid(c, s, voxel_mm=1.0, z_floor=-100.0)
    assert grid.origin[2] == pytest.approx(-100.0)
    vol = deposit_gaussians(center, sigma, np.array([1.0]), energy, kernel, grid)
    inside = float(kernel.mass_fraction(150.0, -100.0, 0.0))
    assert 0.2 < inside < 0.6
    assert float(vol.sum()) == pytest.approx(inside, rel=1e-3)


def test_entrance_wet_takes_range_and_a_little_fluence() -> None:
    from scan_kit.views.dose_volume_physics import WATER, csda_range_mm, through_wet

    e = np.array([100.0, 150.0, 60.0])
    e_in, kept = through_wet(e, 30.0)
    np.testing.assert_allclose(csda_range_mm(WATER, e_in[:2]), csda_range_mm(WATER, e[:2]) - 30.0, atol=0.05)
    # ~1.2 % of primaries per cm of water go to nuclear interactions.
    assert 0.95 < kept[1] < 0.975
    # 60 MeV ranges out at ~31 mm, just past 30 mm; far less makes it with 40 mm.
    assert through_wet([60.0], 40.0)[1][0] == 0.0


def test_auto_depth_for_250_mev_in_water() -> None:
    from scan_kit.views.dose_volume_physics import WATER
    from scan_kit.views.dose_volume_vispy import auto_depth_mm

    # R ≈ 379 mm, σ ≈ 7.5 mm at 1 % spread.
    assert auto_depth_mm(WATER, [70.0, 250.0], 1.0, 5.0) == pytest.approx(417.0, abs=2.0)
    assert auto_depth_mm(WATER, [70.0, 250.0], 1.0, 3.0) == pytest.approx(402.0, abs=2.0)
    assert auto_depth_mm(WATER, [], 1.0, 5.0) == 0.0


def test_field_box_is_the_half_maximum_extent() -> None:
    from scan_kit.views.dose_volume_vispy import field_box

    vol = np.zeros((4, 5, 6), dtype=np.float32)
    vol[1:3, 2:4, 1:4] = 1.0
    vol[2, 3, 2] = 2.0
    origin, extent = field_box(vol, [-10.0, 0.0, -40.0], 1.0)
    np.testing.assert_allclose(origin, [-9.0, 2.0, -39.0])
    np.testing.assert_allclose(extent, [3.0, 2.0, 2.0])
    assert field_box(np.zeros((2, 2, 2)), [0, 0, 0], 1.0) is None
    # Entrance is wide but below half the peak, so the depth edge leaves it out.
    # A slice at 60 % of the peak stays, and its own 50 % sets the width.
    spread = np.zeros((3, 3, 5), dtype=float)
    spread[0, 1, :] = 1.0
    spread[1, 1, :] = 4.0
    spread[1, 1, 2] = 6.0
    spread[2, 1, 2] = 10.0
    _o, wide = field_box(spread, [0, 0, 0], 1.0, per_slice=True)
    assert wide[0] == 5.0 and wide[2] == 2.0
    _o, hot = field_box(spread, [0, 0, 0], 1.0, per_slice=False)
    assert hot[0] == 1.0 and hot[2] == 2.0


def test_phantom_box_spans_surface_to_back_face() -> None:
    from scan_kit.views.dose_volume_vispy import phantom_box

    # Stops mode: the grid starts below the surface, the phantom still starts at 0.
    o, e = phantom_box([-30.0, -20.0, -200.0], [60.0, 40.0, 163.0], 200.0, pad_mm=10.0)
    np.testing.assert_allclose(o, [-40.0, -30.0, -200.0])
    np.testing.assert_allclose(o + e, [40.0, 30.0, 0.0])


def test_phantom_note_reports_depth_and_losses() -> None:
    from scan_kit.views.dose_volume_physics import WATER
    from scan_kit.views.dose_volume_vispy import phantom_note

    auto = phantom_note(WATER, 169.2, 0.0, auto=True, n_nozzle=2, entry_energy=[100.0, 150.0])
    assert auto == "Phantom auto 170 mm water"
    fixed = phantom_note(WATER, 100.0, 20.0, auto=False, n_nozzle=4, entry_energy=[100.0, 150.0])
    assert "Phantom 100 mm water" in fixed
    assert "50 % of spots stop in it" in fixed
    assert "50 % of spots range out past the back" in fixed


def test_gamma_passes_identical_and_fails_scaled() -> None:
    from scan_kit.views.dose_volume_physics import GammaCriteria, gamma_index

    z, y, x = np.mgrid[0:12, 0:12, 0:12].astype(float)
    ref = np.exp(-((x - 6) ** 2 + (y - 6) ** 2 + (z - 6) ** 2) / 8.0)
    crit = GammaCriteria()
    gam, passed, n = gamma_index(ref, ref, 1.0, crit)
    assert n > 0 and passed == n and float(gam.max()) == 0.0
    # 1 mm shift is inside 2 mm DTA; a 10 % hot plan is outside 3 %.
    _g, passed, n = gamma_index(np.roll(ref, 1, axis=2), ref, 1.0, crit)
    assert passed == n
    _g, passed, n = gamma_index(ref * 1.1, ref, 1.0, crit)
    assert passed < n
    # A 0 % cutoff still skips empty voxels, or air would inflate the pass rate.
    sparse = np.where(ref > 0.05, ref, 0.0)
    _g, _passed, n = gamma_index(sparse, sparse, 1.0, GammaCriteria(cutoff_pct=0.0))
    assert n == int((sparse > 0.0).sum())


def test_sequential_scales_start_dark_and_brighten() -> None:
    from scan_kit.views.dose_volume_catalog import DIVERGENT_SCALES, SEQUENTIAL_SCALES
    from scan_kit.views.dose_volume_fill import colormap_samples

    luma = np.array([0.2126, 0.7152, 0.0722])
    for name, _label in SEQUENTIAL_SCALES:
        y = colormap_samples(name, 64) @ luma
        assert y[0] < 0.25, name
        # Turbo is a rainbow and ends on dark red by design.
        assert name == "turbo" or y[-1] > y[0] + 0.5, name
    assert DIVERGENT_SCALES[0][0] == "managua_r"


def test_gamma_verdict_follows_tg218_limits() -> None:
    from scan_kit.views.dose_volume_window import gamma_pass_rate, gamma_verdict

    assert gamma_pass_rate(None) is None
    assert gamma_pass_rate((0, 0)) is None
    assert gamma_pass_rate((95, 100)) == pytest.approx(95.0)
    assert gamma_verdict(95.0)[0] == "Within tolerance"
    assert gamma_verdict(92.0)[0].startswith("Below tolerance")
    assert gamma_verdict(89.9)[0] == "Below action limit"


def test_gpu_layer_fill_and_gamma_match_python(qapp) -> None:
    from scan_kit.views.dose_volume_physics import (
        WATER, GammaCriteria, LayerDoseKernel, build_layer_tables, gamma_index,
    )
    from scan_kit.views.dose_volume_raycast import (
        _alloc_texture, _gl_at_least, _spot_array, gamma_texture, gpu_fill_texture, read_texture,
    )
    from scan_kit.views.vispy_plot import make_scene_canvas

    try:
        canvas = make_scene_canvas(size=(64, 64), show=False, gl="gl+")
        canvas.set_current()
    except Exception as exc:
        pytest.skip(f"visPy canvas unavailable: {exc}")
    if not _gl_at_least(4, 3):
        pytest.skip("OpenGL 4.3 compute unavailable")
    energy = np.array([90.0, 110.0, 110.0])
    kernel = LayerDoseKernel(build_layer_tables(WATER, energy, 1.0, nodes=512))
    center = np.array([[0.0, 0.0, 0.0], [6.0, -4.0, 0.0], [-5.0, 3.0, 0.0]])
    sigma = np.full((3, 3), 3.0)
    weight = np.array([1.0, 2.0, 1.5])
    c, s = kernel.span(center, sigma, energy)
    grid = dose_grid(c, s, voxel_mm=2.0)
    nx, ny, nz = grid.shape
    want = deposit_gaussians(center, sigma, weight, energy, kernel, grid) / grid.voxel**3
    tex = [_alloc_texture((nz, ny, nx)) for _ in range(3)]
    gpu_fill_texture(canvas, tex[0], _spot_array(center, sigma, weight, energy, kernel), grid, kernel.tables)
    got = read_texture(canvas, tex[0], (nz, ny, nx))
    assert np.abs(got - want).max() <= 1e-3 * want.max()

    # Plan is 5 % hotter: GPU pass count and γ must match the Python search.
    tex[1].set_data(np.ascontiguousarray(want * 1.05, dtype=np.float32))
    crit = GammaCriteria()
    passed, n = gamma_texture(canvas, tex[0], tex[1], tex[2], grid, crit)
    gam = read_texture(canvas, tex[2], (nz, ny, nx))
    ref_gam, ref_pass, ref_n = gamma_index(got, want * 1.05, grid.voxel, crit)
    assert n == ref_n and abs(passed - ref_pass) <= max(2, ref_n // 500)
    assert np.abs(gam - ref_gam).max() < 0.02

