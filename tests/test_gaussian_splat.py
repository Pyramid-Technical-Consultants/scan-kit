"""Tests for Gaussian splat data mapping and the visPy visual."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from scan_kit.common.ic_trajectory import IC1_Z_MM, IC2_Z_MM, IC_SEP_MM
from scan_kit.views.gaussian_splat_catalog import (
    AGREE_TRANSPARENT,
    COLOR_MU,
    COLOR_PROTONS,
    ERROR_ABSOLUTE,
    ERROR_PERCENT,
    SplatConfig,
    XY_IC1,
    XY_IC2,
    XY_ISO_RAY,
    XY_PLAN,
)
from scan_kit.views.gaussian_splat_data import (
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
    parse_kmu_c_per_mu,
    protons_from_charge_c,
    protons_from_mu,
    range_axis_for_medium,
)
from scan_kit.views.gaussian_splat_vispy import apply_gantry, axis_guide_points
from scan_kit.views.gaussian_splat_visual import (
    COLD_RGB,
    HOT_RGB,
    auto_amp_scale,
    expand_splat_vertices,
    finite_diff_jacobian,
    project_covariance,
    residual_abs_fbo_scale,
    residual_agreement_rgba,
    residual_error_alpha,
    residual_error_mag,
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


def test_project_covariance_axis_aligned_scale() -> None:
    def map_xy(p):
        return np.array([2.0 * p[0], 3.0 * p[1]])

    jac = finite_diff_jacobian(map_xy, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(jac[:, 0], [2.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(jac[:, 1], [0.0, 3.0], atol=1e-9)
    l1, l2, evecs = project_covariance(np.array([4.0, 1.0, 0.5]), jac)
    # Screen σ = (8, 3); variance (64, 9).
    assert l1 == pytest.approx(64.0)
    assert l2 == pytest.approx(9.0)
    np.testing.assert_allclose(np.abs(evecs[:, 0]), [1.0, 0.0], atol=1e-6)


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


def test_expand_splat_vertices_repeats_instance_attrs() -> None:
    centers = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    sigmas = np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], dtype=np.float32)
    weights = np.array([2.0, 3.0], dtype=np.float32)
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    corners, c, s, w, col = expand_splat_vertices(centers, sigmas, weights, rgb)
    assert corners.shape == (12, 2)
    np.testing.assert_allclose(c[0:6], np.broadcast_to(centers[0], (6, 3)))
    np.testing.assert_allclose(c[6:12], np.broadcast_to(centers[1], (6, 3)))
    np.testing.assert_allclose(w[0:6], 2.0)
    np.testing.assert_allclose(col[6:12], np.broadcast_to(rgb[1], (6, 3)))
    empty_c, empty_s, empty_w, empty_rgb = (
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
    )
    corners0, *_ = expand_splat_vertices(empty_c, empty_s, empty_w, empty_rgb)
    assert corners0.shape == (0, 2)


def test_residual_agreement_zero_is_transparent_or_white() -> None:
    assert SplatConfig().agreement == AGREE_TRANSPARENT
    meas = np.array([1.0, 1.0, 0.0, 0.0])
    plan = np.array([1.0, 0.0, 1.0, 0.0])
    clear = residual_agreement_rgba(meas, plan, zero="transparent")
    np.testing.assert_allclose(clear[0, 3], 0.0, atol=1e-12)
    np.testing.assert_allclose(clear[1, :3], HOT_RGB)
    np.testing.assert_allclose(clear[1, 3], 1.0)
    np.testing.assert_allclose(clear[2, :3], COLD_RGB)
    np.testing.assert_allclose(clear[3, 3], 0.0)
    white = residual_agreement_rgba(meas, plan, zero="white")
    np.testing.assert_allclose(white[0, :3], 1.0)
    np.testing.assert_allclose(white[0, 3], 1.0)
    np.testing.assert_allclose(white[3, 3], 0.0)
    # Relative |m−p|/(m+p) is 1 on a faint tail; absolute |m−p| stays dim.
    tail = residual_agreement_rgba(np.array([0.03]), np.array([0.0]), zero="transparent")
    assert tail[0, 3] < 0.15
    core = residual_agreement_rgba(np.array([1.0]), np.array([0.3]), zero="transparent")
    assert core[0, 3] > 0.5
    mid = residual_error_alpha(0.15)
    assert 0.2 < mid < 0.9
    assert residual_error_alpha(0.0) == 0.0
    assert residual_error_alpha(1.0) == 1.0


def test_residual_error_percent_vs_absolute() -> None:
    assert SplatConfig().error_mode == ERROR_PERCENT
    np.testing.assert_allclose(SplatConfig().error_scale, 0.10)
    np.testing.assert_allclose(
        residual_error_mag(np.array([1.1]), np.array([1.0]), mode=ERROR_PERCENT),
        [0.1],
    )
    np.testing.assert_allclose(
        residual_error_mag(np.array([1.1]), np.array([1.0]), mode=ERROR_ABSOLUTE),
        [0.1],
    )
    sat = residual_agreement_rgba(
        np.array([1.1]), np.array([1.0]),
        zero="transparent", mode=ERROR_PERCENT, scale=0.10,
    )
    assert sat[0, 3] > 0.9
    faint = residual_agreement_rgba(
        np.array([1.01]), np.array([1.0]),
        zero="transparent", mode=ERROR_PERCENT, scale=0.10,
    )
    assert faint[0, 3] < 0.3
    # Faint unmatched tails would be infinite %; hide them (no hollow shell).
    tail = residual_agreement_rgba(
        np.array([0.03]), np.array([0.0]),
        zero="transparent", mode=ERROR_PERCENT, scale=0.10,
    )
    assert tail[0, 3] == 0.0
    np.testing.assert_allclose(
        residual_abs_fbo_scale(np.array([1.0, 1.0]), 0.2, 1.0), 0.2,
    )
    np.testing.assert_allclose(
        residual_abs_fbo_scale(np.array([0.5]), 0.2, 1.0), 0.4,
    )
    np.testing.assert_allclose(
        residual_abs_fbo_scale(np.array([1.0]), 0.2, 0.5), 0.1,
    )


def test_auto_amp_scale_makes_typical_peak_unity() -> None:
    sigmas = np.full((8, 3), 5.0, dtype=float)
    weights = np.ones(8, dtype=float)
    scale = auto_amp_scale(weights, sigmas)
    peak = 1.0 / (2.0 * np.pi * 25.0)
    np.testing.assert_allclose(scale * peak, 1.0, rtol=1e-6)
    neg_scale = auto_amp_scale(-weights, sigmas)
    np.testing.assert_allclose(neg_scale, scale)
    assert auto_amp_scale(np.array([]), np.empty((0, 3))) == 1.0


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
    np.testing.assert_allclose(color_scalar(cloud, SplatConfig(color_mode=COLOR_MU)), [0.5])
    ten = color_scalar(cloud, SplatConfig(color_mode=COLOR_PROTONS, ic_gap_mm=10.0))
    five = color_scalar(cloud, SplatConfig(color_mode=COLOR_PROTONS, ic_gap_mm=5.0))
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


def test_gaussian_splat_module_has_run() -> None:
    from scan_kit.views import gaussian_splat

    assert callable(gaussian_splat.run)


def test_offscreen_splat_render(qapp) -> None:
    from scan_kit.views.gaussian_splat_visual import make_gaussian_splat_node
    from scan_kit.views.vispy_plot import make_scene_canvas

    try:
        canvas = make_scene_canvas(size=(120, 80), show=False, gl="gl+")
    except Exception:
        try:
            canvas = make_scene_canvas(size=(120, 80), show=False)
        except Exception as exc:
            pytest.skip(f"visPy canvas unavailable: {exc}")
    try:
        from vispy import scene

        view = canvas.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(fov=45, distance=50)
        splat = make_gaussian_splat_node()(parent=view.scene)
        n = 4
        centers = np.array(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 2.0], [0.0, 5.0, 4.0], [-3.0, -2.0, 1.0]],
            dtype=np.float32,
        )
        sigmas = np.full((n, 3), 1.5, dtype=np.float32)
        weights = np.ones(n, dtype=np.float32)
        rgb = np.array([[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.4, 1.0], [1.0, 1.0, 0.2]],
                       dtype=np.float32)
        splat.set_data(centers, sigmas, weights, rgb, gain=2.0)
        img = np.asarray(canvas.render())
    except Exception as exc:
        pytest.skip(f"visPy cannot render splats offscreen: {exc}")
    if img.ndim != 3 or img.size == 0:
        pytest.skip("visPy render returned an empty image")
    assert img.shape[0] >= 80
    assert img[..., :3].max() > 0
