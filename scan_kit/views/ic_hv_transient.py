"""IC high-voltage transient safety-test data (``ic_hv_toggle``).

When the IC bias high voltage is stepped (``starting_voltage`` →
``ending_voltage``), charge ``Q = C·ΔV`` flows through each readout channel's
coupling capacitance.  The dose-controller safety test records the resulting
current transients and grades every channel pass/fail.  Each session that ran
the test stores, per IC device, inside ``ic_hv_toggle/``:

* ``<device>_HCC.csv`` — primary high-charge-collector channel current(s) in
  **nA**, sampled every ms.
* ``<device>_Strips.csv`` — per-strip current samples in **nA** (256 channels
  for the strip ICs), captured around the toggle event.
* ``<device>_result.json`` — firmware pass/fail grading.

The applied step and the firmware's expected capacitance live in
``config/nozzle/<device>/config.json`` under ``dose_controller/safety_test/
hv_transient_test/``.

Rather than trust the file's graded values, this view **re-derives the coupling
capacitance from the raw nA waveforms**: it integrates the transient current
over the step window to get the injected charge (``nA·ms = pC``) and divides by
the applied ``ΔV`` to get ``C`` in **pF**, then compares our measurement to the
firmware result.  Sessions are overlaid for comparison like the other views.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..common import (
    DEFAULT_SESSION_COLORS,
    GRID_KW,
    finish_view,
    view_grid,
)
from ..common.session_source import ensure_session_on_disk

_log = logging.getLogger(__name__)

_trapz = getattr(np, "trapezoid", np.trapz)

VIEW_TITLE = "IC HV Transient Test"

_HV_SUBDIR = "ic_hv_toggle"
_HCC_SUFFIX = "_HCC.csv"
_STRIPS_SUFFIX = "_Strips.csv"
_RESULT_SUFFIX = "_result.json"

# Transient-window detection / integration tunables.
_WINDOW_THRESHOLD_FRAC = 0.10  # window edges where the step decays below this × peak
_WINDOW_PAD_MS = 40.0          # extra padding each side of the detected step
_DEFAULT_DELTA_V = 80.0        # fallback applied step (V) if config is unreadable


@dataclass
class DeviceHv:
    """One IC device's measured HV-transient capacitances for a session."""

    device: str
    waveform: pd.DataFrame          # time (ms) + HCC channel currents (nA)
    window: tuple[float, float]     # integration window (ms)
    delta_v: float                  # applied voltage step (V)
    hcc_caps: dict[str, float]      # measured C per HCC channel (pF)
    strip_caps: np.ndarray | None   # measured C per strip channel (pF)
    strip_fail: np.ndarray | None   # firmware per-channel fail mask
    overall: str | None             # firmware pass/fail summary
    n_fail: int                     # firmware failed-channel count
    n_graded: int                   # firmware graded-channel count
    expected: dict[str, float] = field(default_factory=dict)  # config expectations

    @property
    def primary_cap(self) -> float:
        """Capacitance of the primary HCC channel (first column)."""
        return next(iter(self.hcc_caps.values()), float("nan"))

    @property
    def has_strips(self) -> bool:
        return self.strip_caps is not None and self.strip_caps.size > 0


# ── Capacitance reconstruction ────────────────────────────────────────────


def _detect_window(t: np.ndarray, primary: np.ndarray) -> tuple[float, float] | None:
    """Bracket the HV step from the primary channel's positive current excursion.

    The +ΔV step injects positive charge, so the step shows as the largest
    *positive* deviation from baseline; the window extends while the current
    stays above a small fraction of that peak (the RC decay back to baseline).
    """
    base = float(np.median(primary))
    d = primary - base
    i = int(np.argmax(d))
    if not np.isfinite(d[i]) or d[i] <= 0:
        return None
    thr = d[i] * _WINDOW_THRESHOLD_FRAC
    lo = i
    while lo > 0 and d[lo - 1] > thr:
        lo -= 1
    hi = i
    while hi < len(d) - 1 and d[hi + 1] > thr:
        hi += 1
    return float(t[lo] - _WINDOW_PAD_MS), float(t[hi] + _WINDOW_PAD_MS)


def _channel_caps(
    df: pd.DataFrame, t0: float, t1: float, delta_v: float
) -> dict[str, float]:
    """Measured capacitance (pF) per channel = ∫(I − baseline) dt / ΔV.

    Current is nA and time is ms, so ``nA·ms = pC`` and ``pC / V = pF``.  The
    baseline is the median of samples *outside* the transient window.
    """
    t = df.iloc[:, 0].to_numpy(dtype=float)
    in_win = (t >= t0) & (t <= t1)
    out_win = ~in_win
    caps: dict[str, float] = {}
    for col in df.columns[1:]:
        y = df[col].to_numpy(dtype=float)
        base = float(np.median(y[out_win])) if out_win.sum() > 5 else float(np.median(y))
        tt = t[in_win]
        yy = y[in_win] - base
        ok = np.isfinite(tt) & np.isfinite(yy)
        charge_pc = float(_trapz(yy[ok], tt[ok])) if ok.sum() >= 2 else 0.0
        caps[str(col)] = charge_pc / delta_v if delta_v else float("nan")
    return caps


# ── Loading ───────────────────────────────────────────────────────────────


def _read_csv(path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - optional data, never fatal
        _log.debug("Failed reading %s: %s", path, exc)
        return None
    return df if not df.empty and df.shape[1] >= 2 else None


def _flatten_results(obj) -> tuple[list[str], list[str]]:
    """Walk a result JSON, returning (scalar pass/fail leaves, channel array)."""
    scalars: list[str] = []
    channel_array: list[str] = []

    def _walk(node) -> None:
        nonlocal channel_array
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            strs = [str(v).lower() for v in node if isinstance(v, str)]
            if strs and all(s in ("pass", "fail") for s in strs):
                if len(strs) > len(channel_array):
                    channel_array = strs
        elif isinstance(node, str) and node.lower() in ("pass", "fail"):
            scalars.append(node.lower())

    _walk(obj)
    return scalars, channel_array


def _parse_result(text: str | None) -> tuple[str | None, list[str]]:
    """Return (overall pass/fail, per-channel pass/fail list) from result JSON."""
    if not text:
        return None, []
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return None, []
    result = payload.get("result", payload) if isinstance(payload, dict) else payload
    scalars, channels = _flatten_results(result)
    all_flags = scalars + channels
    if not all_flags:
        return None, channels
    overall = "fail" if any(f == "fail" for f in all_flags) else "pass"
    return overall, channels


def _load_hv_config(root, device: str) -> dict[str, float]:
    """Read the applied step (ΔV) and expected capacitances from device config."""
    cfg_path = root / "config" / "nozzle" / device / "config.json"
    out: dict[str, float] = {"delta_v": _DEFAULT_DELTA_V}
    if not cfg_path.is_file():
        return out
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return out
    if not isinstance(cfg, dict):
        return out

    def _find(suffix: str):
        for key, value in cfg.items():
            if isinstance(key, str) and key.endswith(suffix):
                return value
        return None

    start = _find("hv_transient_test/starting_voltage")
    end = _find("hv_transient_test/ending_voltage")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        dv = abs(float(end) - float(start))
        if dv > 0:
            out["delta_v"] = dv
    for name, suffix in (
        ("exp_integrator", "integrator_result/expected_capacitance"),
        ("exp_primary", "hv_transient_primary_channel_test/expected_capacitance"),
        ("exp_channels", "hv_transient_256_channels_test/expected_capacitance"),
        ("tolerance", "integrator_result/capacitance_tolerance"),
    ):
        val = _find(suffix)
        if isinstance(val, (int, float)):
            out[name] = float(val)
    return out


def _load_device(root, hv_dir, device: str) -> DeviceHv | None:
    """Assemble one device's measured HV-transient capacitances."""
    hcc = _read_csv(hv_dir / f"{device}{_HCC_SUFFIX}")
    if hcc is None:
        return None

    strips = _read_csv(hv_dir / f"{device}{_STRIPS_SUFFIX}")
    result_path = hv_dir / f"{device}{_RESULT_SUFFIX}"
    result_text = (
        result_path.read_text(encoding="utf-8") if result_path.is_file() else None
    )
    overall, channel_flags = _parse_result(result_text)

    cfg = _load_hv_config(root, device)
    delta_v = cfg.get("delta_v", _DEFAULT_DELTA_V)

    t = hcc.iloc[:, 0].to_numpy(dtype=float)
    primary = hcc.iloc[:, 1].to_numpy(dtype=float)
    window = _detect_window(t, primary) or (float(t[0]), float(t[-1]))

    hcc_caps = _channel_caps(hcc, window[0], window[1], delta_v)

    strip_caps: np.ndarray | None = None
    if strips is not None:
        strip_caps = np.array(
            list(_channel_caps(strips, window[0], window[1], delta_v).values()),
            dtype=float,
        )

    strip_fail: np.ndarray | None = None
    if strip_caps is not None and len(channel_flags) == strip_caps.size:
        strip_fail = np.array([f == "fail" for f in channel_flags], dtype=bool)

    n_graded = len(channel_flags)
    n_fail = sum(1 for f in channel_flags if f == "fail")

    return DeviceHv(
        device=device,
        waveform=hcc,
        window=window,
        delta_v=delta_v,
        hcc_caps=hcc_caps,
        strip_caps=strip_caps,
        strip_fail=strip_fail,
        overall=overall,
        n_fail=n_fail,
        n_graded=n_graded,
        expected={k: v for k, v in cfg.items() if k != "delta_v"},
    )


def _discover_devices(hv_dir) -> list[str]:
    """Device names with an HCC file present (sorted for stable rows)."""
    return sorted(
        p.name[: -len(_HCC_SUFFIX)]
        for p in hv_dir.glob(f"*{_HCC_SUFFIX}")
        if p.is_file()
    )


def _load_session_hv(session_id: str, base_dir: str) -> dict[str, DeviceHv] | None:
    """Load every IC device's HV-transient data for one session."""
    root = ensure_session_on_disk(session_id, base_dir)
    if root is None:
        return None
    hv_dir = root / _HV_SUBDIR
    if not hv_dir.is_dir():
        return None

    devices: dict[str, DeviceHv] = {}
    for device in _discover_devices(hv_dir):
        data = _load_device(root, hv_dir, device)
        if data is not None:
            devices[device] = data
    return devices or None


# ── Plotting ───────────────────────────────────────────────────────────────


def _expected_for_display(dev: DeviceHv) -> tuple[float | None, float | None]:
    """The config expected capacitance + tolerance most relevant to *dev*."""
    exp = dev.expected
    key = "exp_primary" if dev.has_strips else "exp_integrator"
    return exp.get(key), exp.get("tolerance")


def _result_color(dev: DeviceHv) -> str:
    return "red" if (dev.overall == "fail" or dev.n_fail) else "green"


def _plot_waveform(ax, per_session: dict[str, DeviceHv], colors, single: bool):
    """HCC transient current vs time, with the integration window shaded."""
    for (sid, dev), color in zip(per_session.items(), colors):
        df = dev.waveform
        t = df.iloc[:, 0].to_numpy(dtype=float)
        channel_cols = list(df.columns[1:])
        n_ch = len(channel_cols)
        for ci, col in enumerate(channel_cols):
            y = df[col].to_numpy(dtype=float)
            if single:
                ax.plot(t, y, linewidth=0.9, alpha=0.9, label=col)
            else:
                ax.plot(t, y, color=color, linewidth=0.7,
                        alpha=0.4 + 0.5 * (1.0 if n_ch == 1 else 1 - ci / n_ch))
        t0, t1 = dev.window
        ax.axvspan(t0, t1, color="gray" if single else color, alpha=0.10, lw=0)

    ax.set_xlabel("Time (ms)")
    ax.grid(**GRID_KW)
    if single:
        ax.legend(fontsize=7, loc="upper right", ncol=2)


def _plot_channel_caps(ax, per_session: dict[str, DeviceHv], colors):
    """Measured capacitance per channel (strips, or HCC channels for the quad)."""
    for (sid, dev), color in zip(per_session.items(), colors):
        if dev.has_strips:
            caps = dev.strip_caps
            x = np.arange(1, caps.size + 1)
            ax.plot(x, caps, color=color, linewidth=0.9, alpha=0.85)
            if dev.strip_fail is not None and dev.strip_fail.any():
                ax.scatter(x[dev.strip_fail], caps[dev.strip_fail], color="red",
                           s=10, zorder=5, edgecolors="none", alpha=0.7)
        else:
            caps = np.array(list(dev.hcc_caps.values()), dtype=float)
            x = np.arange(1, caps.size + 1)
            ax.plot(x, caps, color=color, marker="o", markersize=5, linewidth=1.0,
                    alpha=0.85)

    is_strip = any(d.has_strips for d in per_session.values())
    ax.set_xlabel("Strip channel" if is_strip else "HCC channel")
    ax.set_ylabel("Measured C (pF)")
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
    ax.grid(**GRID_KW)


def _measured_metrics(dev: DeviceHv) -> list[tuple[str, float]]:
    """Per-device summary capacitances for the measured-vs-file bars."""
    if dev.has_strips:
        caps = dev.strip_caps[np.isfinite(dev.strip_caps)]
        return [
            ("Primary\nHCC", dev.primary_cap),
            ("Σ strips", float(np.nansum(caps))),
        ]
    return [(c.replace("hcc_", "HCC "), v) for c, v in dev.hcc_caps.items()]


def _plot_measured_vs_file(ax, per_session: dict[str, DeviceHv], colors):
    """Grouped bars of our measured capacitances + the firmware/config result."""
    devs = list(per_session.values())
    sids = list(per_session)
    cats = [lbl for lbl, _ in _measured_metrics(devs[0])]
    n_cat = len(cats)
    n_sess = len(devs)
    width = 0.8 / max(n_sess, 1)
    x = np.arange(n_cat)

    for si, dev in enumerate(devs):
        vals = [v for _, v in _measured_metrics(dev)]
        offset = (si - (n_sess - 1) / 2) * width
        ax.bar(x + offset, vals, width, color=colors[si], alpha=0.85,
               label=sids[si] if n_sess > 1 else None)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("Measured C (pF)")
    ax.grid(axis="y", **GRID_KW)

    lines: list[tuple[str, str]] = []
    for sid, dev in per_session.items():
        exp, tol = _expected_for_display(dev)
        result = dev.overall.upper() if dev.overall else "—"
        if dev.n_graded:
            result = f"{result} ({dev.n_graded - dev.n_fail}/{dev.n_graded})"
        exp_txt = f"file exp {exp:g} pF" if exp is not None else "file exp —"
        if tol is not None:
            exp_txt += f" ±{tol:g}%"
        prefix = f"{sid[:6]}: " if len(per_session) > 1 else ""
        lines.append((f"{prefix}meas {dev.primary_cap:.3g} pF | {exp_txt} | fw {result}",
                      _result_color(dev)))

    for k, (txt, color) in enumerate(lines):
        ax.text(0.5, 0.98 - k * 0.075, txt, transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, color=color, fontweight="bold")


# ── Main entry point ───────────────────────────────────────────────────────

_COLUMNS = (
    ("HV transient current (nA)", _plot_waveform),
    ("Measured capacitance / channel", _plot_channel_caps),
    ("Measured vs file", _plot_measured_vs_file),
)


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Render the IC HV transient test view for the selected session(s)."""
    del settings

    if not session_ids:
        return

    session_data: dict[str, dict[str, DeviceHv]] = {}
    for sid in session_ids:
        data = _load_session_hv(sid, base_dir)
        if data:
            session_data[sid] = data

    if not session_data:
        _log.info("No IC HV transient (ic_hv_toggle) data for the selected sessions")
        return

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    color_by_sid = dict(zip(loaded_ids, colors))
    single = len(loaded_ids) == 1

    devices = sorted({dev for data in session_data.values() for dev in data})
    fig, axes = view_grid(len(devices), len(_COLUMNS), cell_w=5.5, cell_h=3.2)

    for row, device in enumerate(devices):
        per_session = {
            sid: session_data[sid][device]
            for sid in loaded_ids
            if device in session_data[sid]
        }
        dev_colors = [color_by_sid[sid] for sid in per_session]

        _plot_waveform(axes[row, 0], per_session, dev_colors, single)
        _plot_channel_caps(axes[row, 1], per_session, dev_colors)
        _plot_measured_vs_file(axes[row, 2], per_session, dev_colors)

        if row == 0:
            for col, (title, _) in enumerate(_COLUMNS):
                axes[row, col].set_title(title, fontsize=10)
        axes[row, 0].set_ylabel(f"{device}\nIC current (nA)")

    finish_view(fig, VIEW_TITLE, loaded_ids, colors, base_dir=base_dir)
