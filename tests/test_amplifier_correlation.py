"""Tests for amplifier correlation view data loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scan_kit.common.session_source import resolve_session_source
from scan_kit.views.amplifier_correlation import _load_session_samples

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"


def test_load_g3_amplifier_correlation_samples() -> None:
    samples = _load_session_samples("1943968267", str(TEST_DATA))
    assert samples is not None
    assert samples.cmd_x.size > 0
    assert np.isfinite(samples.readback_x).any()
    assert np.isfinite(samples.field_x).any()
    assert np.isfinite(samples.angle_x_mrad).any()


def test_load_g2_amplifier_correlation_samples() -> None:
    samples = _load_session_samples("590658542", str(TEST_DATA))
    assert samples is not None
    assert samples.cmd_x.size > 0
    assert np.isfinite(samples.readback_x).any()
    assert np.isfinite(samples.field_x).any()
    assert np.isfinite(samples.angle_x_mrad).any()


def test_load_older_g3_amplifier_correlation_samples() -> None:
    """Older G3 exports use r_xV/r_xB and r_ic*_spot_position column names."""
    samples = _load_session_samples("1262268206", str(TEST_DATA))
    assert samples is not None
    assert samples.cmd_x.size > 0
    assert np.isfinite(samples.readback_y).any()
    assert np.isfinite(samples.field_x).any()
    assert np.isfinite(samples.angle_x_mrad).any()


def test_g2_readback_linear_fit_is_near_unity() -> None:
    from scan_kit.views.amplifier_correlation import _linear_residuals

    samples = _load_session_samples("590658542", str(TEST_DATA))
    assert samples is not None
    for cmd, rb in (
        (samples.cmd_x, samples.readback_x),
        (samples.cmd_y, samples.readback_y),
    ):
        result = _linear_residuals(cmd, rb)
        assert result is not None
        gain, offset_v, residual = result
        assert 0.95 <= gain <= 1.05
        assert abs(offset_v) < 0.01
        finite = np.isfinite(residual)
        assert np.median(np.abs(residual[finite])) < 0.01


def test_samples_have_finite_momentum() -> None:
    for sid in ("590658542", "1943968267"):
        samples = _load_session_samples(sid, str(TEST_DATA))
        assert samples is not None
        assert samples.momentum.shape == samples.field_x.shape
        assert np.isfinite(samples.momentum).any()
        finite = samples.momentum[np.isfinite(samples.momentum)]
        assert (finite > 0).all()


def test_cubic_fit_recovers_curved_angle_field_relationship() -> None:
    from scan_kit.views.amplifier_correlation import _cubic_fit, _linear_residuals

    field = np.linspace(-25000.0, 25000.0, 2000)
    # arctan-like compression at large field: true angle bends away from linear.
    angle = np.arctan(3.8e-5 * field) * 1000.0

    lin = _linear_residuals(field, angle)
    cub = _cubic_fit(field, angle)
    assert lin is not None and cub is not None

    lin_med = float(np.median(np.abs(lin[2][np.isfinite(lin[2])])))
    cub_med = float(np.median(np.abs(cub.residual[np.isfinite(cub.residual)])))
    assert cub_med < lin_med
    # Near-zero slope recovers the true small-signal gain (38 mrad / 1000 G).
    assert abs(cub.c1 - 0.038) < 0.003
    # The cubic term is negative (arctan compression at large field).
    assert cub.c3 < 0.0


def test_energy_mev_momentum_roundtrip() -> None:
    from scan_kit.views.amplifier_correlation import (
        _energy_mev_from_momentum,
        _momentum_mev,
    )

    energies = np.array([250.0, 400.0, 600.0])
    recovered = _energy_mev_from_momentum(_momentum_mev(energies))
    np.testing.assert_allclose(recovered, energies, rtol=1e-9)


def test_rigidity_correction_flattens_residual_energy_slope() -> None:
    from scan_kit.views.amplifier_correlation import (
        _REFERENCE_MOMENTUM_MEV,
        _cmd_arc_fit,
        _momentum_mev,
        _residual_energy_slope,
    )

    cmd = np.linspace(-3.0, 3.0, 1200)
    energies = np.repeat([300.0, 500.0, 700.0], 400)
    momentum = _momentum_mev(energies)
    eccmd = cmd * _REFERENCE_MOMENTUM_MEV / momentum
    angle = np.arcsin(0.05 * eccmd / 1000.0) * 1000.0

    raw_fit = _cmd_arc_fit(cmd, angle)
    ec_fit = _cmd_arc_fit(eccmd, angle)
    assert raw_fit is not None and ec_fit is not None

    raw_slope = _residual_energy_slope(
        energies, angle - raw_fit.predict(cmd)
    )
    ec_slope = _residual_energy_slope(
        energies, angle - ec_fit.predict(eccmd)
    )
    assert raw_slope is not None and ec_slope is not None
    assert abs(raw_slope) > 5 * abs(ec_slope)
    assert abs(ec_slope) < 0.01


def test_cmd_arc_fit_recovers_linear_sin_theta_gain() -> None:
    from scan_kit.views.amplifier_correlation import _cmd_arc_fit

    cmd = np.linspace(-3.0, 3.0, 500)
    angle = np.arcsin(0.05 * cmd) * 1000.0
    fit = _cmd_arc_fit(cmd, angle)
    assert fit is not None
    assert abs(fit.c1 - 50.0) < 1.0
    assert abs(fit.c0) < 0.5
    med = float(np.median(np.abs(fit.residual[np.isfinite(fit.residual)])))
    assert med < 0.05


def test_pooled_super_fit_produces_combined_entries() -> None:
    from scan_kit.views.amplifier_correlation import (
        AmplifierCorrelationSamples,
        _PooledRow,
        _format_angle_fit_label,
    )

    rng = np.random.default_rng(0)

    def make(gain_v: float, field_gain: float) -> AmplifierCorrelationSamples:
        cmd = np.linspace(-3.0, 3.0, 400)
        rb = cmd * gain_v + rng.normal(0, 0.01, cmd.size)
        field = rb * field_gain
        momentum = np.full(cmd.size, 1100.0)
        angle = np.arctan(3.8e-5 * field) * 1000.0
        return AmplifierCorrelationSamples(
            cmd_x=cmd, cmd_y=cmd,
            readback_x=rb, readback_y=rb,
            field_x=field, field_y=field,
            angle_x_mrad=angle, angle_y_mrad=angle,
            momentum=momentum,
        )

    pooled = _PooledRow()
    pooled.add(make(1.0, 3000.0), "cmd_x", "readback_x", "field_x", "angle_x_mrad")
    pooled.add(make(1.02, 2900.0), "cmd_x", "readback_x", "field_x", "angle_x_mrad")

    rb_fit = pooled.readback_fit()
    field_fit = pooled.field_fit()
    angle_fit = pooled.angle_fit()
    cmd_arc_fit = pooled.cmd_arc_fit()
    assert rb_fit is not None and field_fit is not None and angle_fit is not None
    assert cmd_arc_fit is not None

    # Pooled cmd→readback gain ~1, field gain order kG/V, cubic recovered.
    assert abs(rb_fit.slope - 1.0) < 0.1
    assert angle_fit.c3 < 0.0
    assert cmd_arc_fit.c1 > 0.0
    label = _format_angle_fit_label(angle_fit, prefix="All: ")
    assert label.startswith("All: ")
    assert "cubic" in label and "mrad/kG" in label and "\u00b5rad/kG" in label


def test_readback_fit_handles_constant_cmd_and_readback() -> None:
    """G3 sessions can hold the beam at fixed amplifier setpoints."""
    from scan_kit.views.amplifier_correlation import _PooledRow

    samples = _load_session_samples("845596095", str(TEST_DATA))
    if samples is None:
        return

    pooled = _PooledRow()
    pooled.add(samples, "cmd_x", "readback_x", "field_x", "angle_x_mrad")
    fit = pooled.readback_fit()
    assert fit is not None
    assert fit.slope == 0.0
    assert np.isfinite(fit.intercept)


def test_energy_correction_reduces_g3_angle_residual() -> None:
    from scan_kit.common import fit_trend
    from scan_kit.views.amplifier_correlation import (
        _energy_corrected_field,
        _linear_residuals,
    )

    samples = _load_session_samples("1943968267", str(TEST_DATA))
    assert samples is not None

    raw_fit = fit_trend(samples.field_x, samples.angle_x_mrad)
    assert raw_fit is not None
    raw_resid = samples.angle_x_mrad - raw_fit.eval(samples.field_x)
    raw_med = float(np.median(np.abs(raw_resid[np.isfinite(raw_resid)])))

    ecfield = _energy_corrected_field(samples.field_x, samples.momentum)
    corrected = _linear_residuals(ecfield, samples.angle_x_mrad)
    assert corrected is not None
    _, _, corr_resid = corrected
    corr_med = float(np.median(np.abs(corr_resid[np.isfinite(corr_resid)])))

    assert corr_med < raw_med


def test_g2_field_physical_scale_is_order_kilogauss_per_volt() -> None:
    from scan_kit.views.amplifier_correlation import _g2_field_physical_scale

    src = resolve_session_source("590658542", str(TEST_DATA))
    assert src is not None
    scale = _g2_field_physical_scale(src)
    assert 500 <= scale <= 5000


def test_g2_field_vs_readback_linear_fit_is_order_kilogauss_per_volt() -> None:
    from scan_kit.views.amplifier_correlation import _linear_residuals

    samples = _load_session_samples("590658542", str(TEST_DATA))
    assert samples is not None
    for rb, field in (
        (samples.readback_x, samples.field_x),
        (samples.readback_y, samples.field_y),
    ):
        result = _linear_residuals(rb, field)
        assert result is not None
        gain, offset_g, residual = result
        assert 500 <= gain <= 5000
        assert abs(offset_g) < 500
        finite = np.isfinite(residual)
        assert np.median(np.abs(residual[finite])) < 500


def test_g3_old_session_readback_tracks_and_field_gain() -> None:
    from scan_kit.common.amplifier_settling import amplifier_readback_tracks_command_axis
    from scan_kit.views.amplifier_correlation import _linear_residuals

    samples = _load_session_samples("863788396", str(TEST_DATA))
    assert samples is not None
    assert amplifier_readback_tracks_command_axis(samples.cmd_x, samples.readback_x)
    result = _linear_residuals(samples.readback_x, samples.field_x)
    assert result is not None
    gain, _, residual = result
    assert 2500 <= gain <= 4500
    assert len(samples.readback_x) > 2000
    assert np.median(np.abs(residual[np.isfinite(residual)])) < 150


def test_g3_stuck_readback_falls_back_to_cmd_for_field() -> None:
    from scan_kit.common.amplifier_settling import amplifier_readback_tracks_command_axis

    samples = _load_session_samples("1943968267", str(TEST_DATA))
    assert samples is not None
    assert not amplifier_readback_tracks_command_axis(samples.cmd_x, samples.readback_x)


def test_settled_samples_are_subset_of_beam_on() -> None:
    from scan_kit.common import detect_beam_on_mask
    from scan_kit.common.amplifier_settling import amplifier_settled_mask
    from scan_kit.common.schema import (
        C_AMPLIFIER_CMD_X,
        C_AMPLIFIER_CMD_Y,
        C_AMPLIFIER_READBACK_X,
        C_AMPLIFIER_READBACK_Y,
        C_MAG_FIELD_X,
        C_MAG_FIELD_Y,
    )
    from scan_kit.common.session_source import load_session_timeslice_device_units
    from scan_kit.views.amplifier_correlation import _g2_field_physical_scale
    from scan_kit.views.timeslice_replay_common import resolve_col

    src = resolve_session_source("590658542", str(TEST_DATA))
    assert src is not None
    frames = load_session_timeslice_device_units(src)
    df = frames[0]
    cx = resolve_col(df.columns, C_AMPLIFIER_CMD_X)
    cy = resolve_col(df.columns, C_AMPLIFIER_CMD_Y)
    rx = resolve_col(df.columns, C_AMPLIFIER_READBACK_X)
    ry = resolve_col(df.columns, C_AMPLIFIER_READBACK_Y)
    beam = detect_beam_on_mask(df)
    assert beam is not None
    cmd_x = df[cx].values.astype(float)
    cmd_y = df[cy].values.astype(float)
    rb_x = df[rx].values.astype(float)
    rb_y = df[ry].values.astype(float)
    bx = resolve_col(df.columns, C_MAG_FIELD_X)
    by = resolve_col(df.columns, C_MAG_FIELD_Y)
    field_scale = _g2_field_physical_scale(src)
    field_x = df[bx].values.astype(float) * field_scale
    field_y = df[by].values.astype(float) * field_scale
    settled = amplifier_settled_mask(
        cmd_x,
        cmd_y,
        rb_x,
        rb_y,
        field_x=field_x,
        field_y=field_y,
        readback_field_drive=True,
    )
    settled_beam = settled & beam
    assert settled_beam.sum() <= beam.sum()
    assert settled_beam.sum() < beam.sum()
    assert settled_beam.sum() > 1000
