"""Tests for amplifier command settling masks."""

from __future__ import annotations

import numpy as np

from scan_kit.common.amplifier_settling import (
    amplifier_command_settled_mask,
    amplifier_readback_settled_mask,
    amplifier_settled_mask,
)


def test_command_settled_mask_excludes_post_step_transient() -> None:
    cmd_x = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    cmd_y = np.zeros_like(cmd_x)
    settled = amplifier_command_settled_mask(cmd_x, cmd_y, settle_samples=3)
    assert settled.tolist() == [False, False, False, False, True, False, False, False, True, True, True]


def test_readback_mask_skips_when_readback_does_not_track() -> None:
    cmd = np.array([1.0, 1.1, 1.2])
    rb = np.array([12.0, 12.0, 12.0])
    assert amplifier_readback_settled_mask(cmd, cmd, rb, rb).all()


def test_readback_mask_applies_when_readback_tracks() -> None:
    cmd = np.array([1.0, 1.0, 1.0])
    rb = np.array([1.0, 1.02, 1.2])
    mask = amplifier_readback_settled_mask(cmd, cmd, rb, rb, tol_v=0.05)
    assert mask.tolist() == [True, True, False]


def test_variance_mask_keeps_stable_plateau() -> None:
    n = 50
    cmd_x = np.ones(n)
    cmd_y = np.zeros(n)
    field_x = np.full(n, 1450.0)
    field_y = np.zeros(n)
    rb_x = cmd_x.copy()
    rb_y = cmd_y.copy()
    mask = amplifier_settled_mask(
        cmd_x, cmd_y, rb_x, rb_y, field_x=field_x, field_y=field_y, variance_window=10
    )
    assert mask[15:45].all()


def test_variance_mask_rejects_cmd_field_ramp_mismatch() -> None:
    """Exclude samples where command moves but field is still at the old level."""
    n = 80
    cmd_x = np.zeros(n)
    cmd_x[10:50] = np.linspace(0.0, 2.0, 40)
    cmd_y = np.zeros(n)
    field_x = np.zeros(n)
    field_x[35:70] = 2900.0
    field_y = np.zeros(n)
    rb_x = cmd_x.copy()
    rb_y = cmd_y.copy()
    mask = amplifier_settled_mask(
        cmd_x, cmd_y, rb_x, rb_y, field_x=field_x, field_y=field_y, variance_window=10
    )
    assert not mask[20:32].any()
    assert mask[50:65].any()


def test_variance_mask_rejects_readback_step() -> None:
    from scan_kit.common.amplifier_settling import _variance_settled_mask

    cmd_x = np.array([1.0] * 20)
    cmd_y = np.zeros_like(cmd_x)
    field_x = np.full(20, 1450.0)
    field_y = np.zeros_like(cmd_x)
    rb_x = np.ones(20)
    rb_x[10:13] = [1.0, 1.15, 1.02]
    rb_x[13:] = 1.02
    rb_y = np.zeros_like(cmd_x)
    mask = _variance_settled_mask(
        cmd_x,
        cmd_y,
        rb_x,
        rb_y,
        field_x=field_x,
        field_y=field_y,
        window=5,
        quantized_field=True,
    )
    assert mask[4:9].all()
    assert not mask[11:14].any()
    assert mask[17]


def test_readback_field_drive_keeps_more_g3_beam_on_samples() -> None:
    """Relaxed readback-driven gates retain more G3 samples when readback tracks."""
    from scan_kit.common import detect_beam_on_mask
    from scan_kit.common.schema import (
        C_AMPLIFIER_CMD_X,
        C_AMPLIFIER_CMD_Y,
        C_AMPLIFIER_READBACK_X,
        C_AMPLIFIER_READBACK_Y,
        C_MAG_FIELD_X,
        C_MAG_FIELD_Y,
    )
    from scan_kit.common.session_source import (
        load_session_timeslice_device_units,
        resolve_session_source,
    )
    from scan_kit.views.timeslice_replay_common import resolve_col

    src = resolve_session_source("863788396", "test_data")
    assert src is not None
    frames = load_session_timeslice_device_units(src)
    strict_total = relaxed_total = 0
    for df in frames:
        beam = detect_beam_on_mask(df)
        if beam is None:
            continue
        cx = resolve_col(df.columns, C_AMPLIFIER_CMD_X)
        cy = resolve_col(df.columns, C_AMPLIFIER_CMD_Y)
        rx = resolve_col(df.columns, C_AMPLIFIER_READBACK_X)
        ry = resolve_col(df.columns, C_AMPLIFIER_READBACK_Y)
        bx = resolve_col(df.columns, C_MAG_FIELD_X)
        by = resolve_col(df.columns, C_MAG_FIELD_Y)
        cmd_x = df[cx].values.astype(float)
        cmd_y = df[cy].values.astype(float)
        rb_x = df[rx].values.astype(float)
        rb_y = df[ry].values.astype(float)
        field_x = df[bx].values.astype(float)
        field_y = df[by].values.astype(float)
        strict = beam & amplifier_settled_mask(
            cmd_x, cmd_y, rb_x, rb_y, field_x=field_x, field_y=field_y
        )
        relaxed = beam & amplifier_settled_mask(
            cmd_x,
            cmd_y,
            rb_x,
            rb_y,
            field_x=field_x,
            field_y=field_y,
            readback_field_drive=True,
        )
        strict_total += int(strict.sum())
        relaxed_total += int(relaxed.sum())
    assert relaxed_total > strict_total
