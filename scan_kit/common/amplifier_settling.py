"""Detect settled amplifier-command plateaus in 1 kHz timeslice data."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Timeslice period is 1 ms → one sample per millisecond.
DEFAULT_VARIANCE_WINDOW_MS = 10
_CMD_CHANGE_TOL = 1e-6
# G2 correcting-coil command ramps in ~50 mV micro-steps; treat 50 mV as the
# effective setpoint step so settling tracks field setpoint updates, not ramp dither.
_QUANTIZED_CMD_CHANGE_TOL_V = 0.05
_FIELD_CHANGE_TOL_G = 0.05
_FIELD_QUANTIZED_P95_MAX_G = 0.5
_READBACK_TRACKING_MAX_MEDIAN_ERR_V = 0.5
_READBACK_SETTLE_TOL_V = 0.05
# Rolling peak-to-peak limits for variance-based transient rejection.
_G2_CMD_FIELD_RESIDUAL_RANGE_MAX_G = 20.0
_G2_READBACK_FIELD_RESIDUAL_RANGE_MAX_G = 40.0
_G2_READBACK_ERR_RANGE_MAX_V = 0.05
_G2_READBACK_DRIVE_RB_ERR_RANGE_MAX_V = 0.10
_G3_FIELD_RANGE_FACTOR = 4.0
_G3_FIELD_RANGE_FACTOR_RELAXED = 6.0
_G3_FIELD_RANGE_MIN_G = 10.0
_G3_READBACK_ERR_RANGE_MAX_V = 1.0
_READBACK_SETTLE_TOL_V_RELAXED = 0.10
_MIN_GAIN_FIT_SAMPLES = 100


def _rolling_peak_to_peak(signal: np.ndarray, window: int) -> np.ndarray:
    """Peak-to-peak range of *signal* over a trailing *window* (milliseconds)."""
    x = np.asarray(signal, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    if window <= 1:
        if n:
            out[:] = 0.0
        return out
    if n < window:
        return out

    views = sliding_window_view(x, window)
    valid = np.all(np.isfinite(views), axis=1)
    ptp = np.ptp(views, axis=1)
    ptp = ptp.astype(float)
    ptp[~valid] = np.nan
    out[window - 1 :] = ptp
    return out


def _step_settled_mask(
    *signals: np.ndarray,
    settle_samples: int,
    change_tols: tuple[float, ...],
) -> np.ndarray:
    """True when sample age is at least *settle_samples* after any signal step."""
    if not signals:
        return np.zeros(0, dtype=bool)
    n = len(signals[0])
    if n == 0:
        return np.zeros(0, dtype=bool)
    if settle_samples <= 0:
        return np.ones(n, dtype=bool)

    changed = np.zeros(n, dtype=bool)
    for sig, tol in zip(signals, change_tols):
        s = sig.astype(float)
        if n > 1:
            changed[1:] |= np.abs(np.diff(s)) > tol

    segment_id = np.cumsum(changed)
    _, segment_starts = np.unique(segment_id, return_index=True)
    age = np.arange(n, dtype=int) - segment_starts[segment_id]
    return age >= settle_samples


def _field_is_quantized(field_x: np.ndarray, field_y: np.ndarray) -> bool:
    """True when field setpoint steps discretely (G2), not continuously (G3 probe)."""
    fx = field_x.astype(float)
    fy = field_y.astype(float)
    if len(fx) < 2:
        return True
    diffs = np.maximum(np.abs(np.diff(fx)), np.abs(np.diff(fy)))
    return float(np.percentile(diffs, 95)) < _FIELD_QUANTIZED_P95_MAX_G


def _estimate_drive_field_gain(
    drive: np.ndarray,
    field: np.ndarray,
    *,
    fit_mask: np.ndarray,
    window: int,
    residual_range_max: float,
) -> float | None:
    """Robust drive→field gain for residual-range transient detection."""
    ok = fit_mask & np.isfinite(drive) & np.isfinite(field) & (np.abs(drive) > 0.05)
    min_fit = min(_MIN_GAIN_FIT_SAMPLES, max(20, len(drive) // 4))
    if int(ok.sum()) < min_fit:
        return None

    drive_span = float(np.ptp(drive[ok]))
    if drive_span < 0.01:
        return None

    gain = float(np.polyfit(drive[ok], field[ok], 1)[0])
    if not np.isfinite(gain) or abs(gain) < 1e-6:
        return None

    residual = field - drive * gain
    stable = ok & (_rolling_peak_to_peak(residual, window) <= residual_range_max)
    if int(stable.sum()) >= min_fit:
        gain = float(np.polyfit(drive[stable], field[stable], 1)[0])
        if not np.isfinite(gain) or abs(gain) < 1e-6:
            return None
    return gain


def _axis_stability_range(
    drive: np.ndarray,
    field: np.ndarray,
    *,
    fit_mask: np.ndarray,
    window: int,
    gain: float | None,
    field_range_max: float,
) -> np.ndarray | None:
    """Rolling range used to detect transients on one axis."""
    active = fit_mask & np.isfinite(drive) & np.isfinite(field) & (np.abs(drive) > 0.05)
    if not active.any():
        return None
    if gain is not None:
        return _rolling_peak_to_peak(field - drive * gain, window)
    return _rolling_peak_to_peak(field, window)


def _variance_settled_mask(
    cmd_x: np.ndarray,
    cmd_y: np.ndarray,
    readback_x: np.ndarray,
    readback_y: np.ndarray,
    *,
    field_x: np.ndarray | None,
    field_y: np.ndarray | None,
    window: int,
    quantized_field: bool,
    readback_field_drive: bool = False,
) -> np.ndarray:
    """True where cmd/field/readback are locally stable (low rolling range)."""
    cmd_x = cmd_x.astype(float)
    cmd_y = cmd_y.astype(float)
    rb_x = readback_x.astype(float)
    rb_y = readback_y.astype(float)
    n = len(cmd_x)
    if n == 0:
        return np.zeros(0, dtype=bool)

    finite = (
        np.isfinite(cmd_x)
        & np.isfinite(cmd_y)
        & np.isfinite(rb_x)
        & np.isfinite(rb_y)
    )
    if field_x is not None and field_y is not None:
        finite &= np.isfinite(field_x) & np.isfinite(field_y)

    rb_err = np.maximum(np.abs(rb_x - cmd_x), np.abs(rb_y - cmd_y))
    rb_range = _rolling_peak_to_peak(rb_err, window)
    mask = finite & np.isfinite(rb_range)
    rb_err_max = (
        _G2_READBACK_DRIVE_RB_ERR_RANGE_MAX_V
        if readback_field_drive
        else _G2_READBACK_ERR_RANGE_MAX_V
    )

    if quantized_field:
        assert field_x is not None and field_y is not None
        field_x = field_x.astype(float)
        field_y = field_y.astype(float)
        drive_x, drive_y = cmd_x, cmd_y
        residual_max = (
            _G2_READBACK_FIELD_RESIDUAL_RANGE_MAX_G
            if readback_field_drive
            else _G2_CMD_FIELD_RESIDUAL_RANGE_MAX_G
        )
        gain_x = _estimate_drive_field_gain(
            drive_x,
            field_x,
            fit_mask=finite,
            window=window,
            residual_range_max=residual_max,
        )
        gain_y = _estimate_drive_field_gain(
            drive_y,
            field_y,
            fit_mask=finite,
            window=window,
            residual_range_max=residual_max,
        )
        signal_ranges: list[np.ndarray] = []
        for drive, field, gain in (
            (drive_x, field_x, gain_x),
            (drive_y, field_y, gain_y),
        ):
            axis_range = _axis_stability_range(
                drive,
                field,
                fit_mask=finite,
                window=window,
                gain=gain,
                field_range_max=_FIELD_CHANGE_TOL_G,
            )
            if axis_range is not None:
                signal_ranges.append(axis_range)
        if not signal_ranges:
            return np.zeros(n, dtype=bool)
        stacked = np.stack(signal_ranges, axis=0)
        signal_range = np.full(n, np.nan)
        with np.errstate(all="ignore"):
            signal_range = np.nanmax(stacked, axis=0)
        range_max = residual_max
        if gain_x is None and gain_y is None:
            range_max = _FIELD_CHANGE_TOL_G
        mask &= np.isfinite(signal_range)
        mask &= signal_range <= range_max
        mask &= rb_range <= rb_err_max
        return mask

    if field_x is not None and field_y is not None:
        field_x = field_x.astype(float)
        field_y = field_y.astype(float)
        field_range = np.maximum(
            _rolling_peak_to_peak(field_x, window),
            _rolling_peak_to_peak(field_y, window),
        )
        mask &= np.isfinite(field_range)
        baseline = field_range[mask]
        range_factor = (
            _G3_FIELD_RANGE_FACTOR_RELAXED
            if readback_field_drive
            else _G3_FIELD_RANGE_FACTOR
        )
        if baseline.size:
            field_max = max(
                _G3_FIELD_RANGE_MIN_G,
                float(np.percentile(baseline, 20)) * range_factor,
            )
        else:
            field_max = _G3_FIELD_RANGE_MIN_G
        mask &= field_range <= field_max

        if readback_field_drive:
            for drive, field in ((rb_x, field_x), (rb_y, field_y)):
                gain = _estimate_drive_field_gain(
                    drive,
                    field,
                    fit_mask=finite,
                    window=window,
                    residual_range_max=_G2_READBACK_FIELD_RESIDUAL_RANGE_MAX_G,
                )
                rb_field_range = _axis_stability_range(
                    drive,
                    field,
                    fit_mask=finite,
                    window=window,
                    gain=gain,
                    field_range_max=_FIELD_CHANGE_TOL_G,
                )
                if rb_field_range is not None:
                    mask &= rb_field_range <= _G2_READBACK_FIELD_RESIDUAL_RANGE_MAX_G

        mask &= rb_range <= _G3_READBACK_ERR_RANGE_MAX_V
        return mask

    mask &= rb_range <= rb_err_max
    return mask


def amplifier_command_settled_mask(
    cmd_x: np.ndarray,
    cmd_y: np.ndarray,
    *,
    settle_samples: int = DEFAULT_VARIANCE_WINDOW_MS,
    cmd_change_tol: float = _CMD_CHANGE_TOL,
) -> np.ndarray:
    """Return True for samples at least *settle_samples* after the last cmd step.

    Command steps are detected when either axis changes by more than
    *cmd_change_tol* between consecutive 1 ms timeslices.
    """
    return _step_settled_mask(
        cmd_x,
        cmd_y,
        settle_samples=settle_samples,
        change_tols=(cmd_change_tol, cmd_change_tol),
    )


def amplifier_readback_settled_mask(
    cmd_x: np.ndarray,
    cmd_y: np.ndarray,
    readback_x: np.ndarray,
    readback_y: np.ndarray,
    *,
    tol_v: float = _READBACK_SETTLE_TOL_V,
    tracking_max_median_err_v: float = _READBACK_TRACKING_MAX_MEDIAN_ERR_V,
) -> np.ndarray:
    """Return True where readback is within *tol_v* of command.

    When median |readback − command| is large (for example G3 sessions where
    readback does not track command), every sample is treated as settled so
    callers can rely on command-step masking alone.
    """
    cmd_x = cmd_x.astype(float)
    cmd_y = cmd_y.astype(float)
    rb_x = readback_x.astype(float)
    rb_y = readback_y.astype(float)
    finite = (
        np.isfinite(cmd_x)
        & np.isfinite(cmd_y)
        & np.isfinite(rb_x)
        & np.isfinite(rb_y)
    )
    if not finite.any():
        return np.ones(len(cmd_x), dtype=bool)

    err = np.maximum(np.abs(rb_x[finite] - cmd_x[finite]), np.abs(rb_y[finite] - cmd_y[finite]))
    if float(np.median(err)) > tracking_max_median_err_v:
        return np.ones(len(cmd_x), dtype=bool)

    return (np.abs(rb_x - cmd_x) <= tol_v) & (np.abs(rb_y - cmd_y) <= tol_v)


def amplifier_readback_tracks_command(
    cmd_x: np.ndarray,
    cmd_y: np.ndarray,
    readback_x: np.ndarray,
    readback_y: np.ndarray,
    *,
    tracking_max_median_err_v: float = _READBACK_TRACKING_MAX_MEDIAN_ERR_V,
) -> bool:
    """True when readback follows command (not a stuck/faulted readback channel)."""
    cmd_x = cmd_x.astype(float)
    cmd_y = cmd_y.astype(float)
    rb_x = readback_x.astype(float)
    rb_y = readback_y.astype(float)
    finite = (
        np.isfinite(cmd_x)
        & np.isfinite(cmd_y)
        & np.isfinite(rb_x)
        & np.isfinite(rb_y)
    )
    if not finite.any():
        return True
    err = np.maximum(np.abs(rb_x[finite] - cmd_x[finite]), np.abs(rb_y[finite] - cmd_y[finite]))
    return float(np.median(err)) <= tracking_max_median_err_v


def amplifier_readback_tracks_command_axis(
    cmd: np.ndarray,
    readback: np.ndarray,
    *,
    tracking_max_median_err_v: float = _READBACK_TRACKING_MAX_MEDIAN_ERR_V,
) -> bool:
    """Per-axis readback tracking check."""
    cmd = cmd.astype(float)
    readback = readback.astype(float)
    finite = np.isfinite(cmd) & np.isfinite(readback)
    if not finite.any():
        return True
    return float(np.median(np.abs(readback[finite] - cmd[finite]))) <= tracking_max_median_err_v


def amplifier_settled_mask(
    cmd_x: np.ndarray,
    cmd_y: np.ndarray,
    readback_x: np.ndarray,
    readback_y: np.ndarray,
    *,
    field_x: np.ndarray | None = None,
    field_y: np.ndarray | None = None,
    variance_window: int = DEFAULT_VARIANCE_WINDOW_MS,
    readback_field_drive: bool = False,
) -> np.ndarray:
    """Combined settled mask: reject locally unstable samples and readback glitches."""
    quantized_field = (
        field_x is not None
        and field_y is not None
        and _field_is_quantized(field_x, field_y)
    )
    rb_tol = (
        _READBACK_SETTLE_TOL_V_RELAXED
        if readback_field_drive
        else _READBACK_SETTLE_TOL_V
    )
    return _variance_settled_mask(
        cmd_x,
        cmd_y,
        readback_x,
        readback_y,
        field_x=field_x,
        field_y=field_y,
        window=variance_window,
        quantized_field=quantized_field,
        readback_field_drive=readback_field_drive,
    ) & amplifier_readback_settled_mask(
        cmd_x, cmd_y, readback_x, readback_y, tol_v=rb_tol
    )
