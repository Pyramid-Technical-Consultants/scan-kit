"""Beam-off ramp-down curve analysis (IC1, IC2, IC3) from scan-total dose."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullFormatter

from ..common import (
    C_LAYER_ID,
    DEFAULT_SESSION_COLORS,
    finish_view,
    GRID_KW,
    make_trend_legend,
    REFLINE_KW,
    trend_line_color,
    trend_session_prefix,
)
from ..common import sliding_background
from ..common.session_source import (
    load_session_csv,
    load_session_timeslice_device_units,
    resolve_session_source,
)
from .timeslice_replay_common import (
    MS_PER_SLICE,
    build_energy_lookups,
    ic_rampdown_signal,
    resolve_col,
    resolve_frame_energy,
    resolve_ic_rampdown_source,
)

import logging

_log = logging.getLogger(__name__)


def _symlog_pct_tick(value: float, _pos: int) -> str:
    """Format symlog axis values as plain percent labels (1%, 10%, 100%, …)."""
    if not np.isfinite(value) or abs(value) < 1e-12:
        return "0%"
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}%"
    if abs(value) >= 1:
        return f"{value:.1f}%"
    return f"{value:g}%"


def _heatmap_energy_extent(energies: list[float] | np.ndarray) -> tuple[float, float]:
    """Return imshow Y extent with padding when energy span is zero."""
    y_lo = float(energies[0])
    y_hi = float(energies[-1])
    if y_hi > y_lo:
        return y_lo, y_hi
    return y_lo - 0.5, y_hi + 0.5


# ---- Tweakable window parameters ------------------------------------------
PRE_OFF_SLICES = 2  # timeslices shown before the falling edge
POST_OFF_SLICES = 9  # timeslices shown after the falling edge
MIN_ON_SLICES = 2  # minimum consecutive above-threshold slices right
# before the falling edge (filters brief spikes)
THRESHOLD_FRAC = 0.10  # fraction of (peak − background) used to define the
# beam-on / beam-off boundary on dose-derived IC signal
MIN_SIGNAL_SPAN = 1e-4  # minimum peak−background (nA or dose-derived rate)
SYMLOG_LINTHRESH = 1.0  # row-0 symlog linear region around 0 (%)
FIT_LINE_KW = dict(linewidth=1.5, linestyle=":", solid_capstyle="round")
# ---------------------------------------------------------------------------

_sliding_background = sliding_background  # local alias for existing call sites


def detect_beam_off_edges(
    signal: np.ndarray,
    threshold_frac: float = THRESHOLD_FRAC,
    min_on_slices: int = MIN_ON_SLICES,
    post_off_slices: int = POST_OFF_SLICES,
) -> np.ndarray:
    """Return sample indices where beam-off ramp-down edges are detected.

    Each returned index is the first *below-threshold* sample after a
    qualifying falling edge (i.e. the first "off" slice).  The same
    validation rules used by ``_extract_windows`` apply:
    * at least *min_on_slices* consecutive beam-on slices before the edge,
    * the signal stays below the threshold for *post_off_slices* after.

    This is a public utility for overlaying edge markers on other views.
    """
    clean = signal
    if np.isnan(signal).any():
        clean = signal.copy()
        clean[np.isnan(clean)] = np.nanmedian(clean)

    bg_array, bg_global, peak = _sliding_background(clean, threshold_frac)
    if peak - bg_global < MIN_SIGNAL_SPAN:
        return np.array([], dtype=int)

    sig = clean - bg_array
    thresh = threshold_frac * (peak - bg_global)
    beam_on = sig > thresh

    diff = np.diff(beam_on.astype(np.int8))
    candidates = np.where(diff == -1)[0] + 1
    n = len(sig)

    edges: list[int] = []
    for idx in candidates:
        anchor = idx - 1
        end = idx + post_off_slices
        if anchor - min_on_slices + 1 < 0 or end > n:
            continue
        if not np.all(beam_on[anchor - min_on_slices + 1 : idx]):
            continue
        if np.any(beam_on[idx:end]):
            continue
        edges.append(idx)
    return np.asarray(edges, dtype=int)


def _extract_windows(
    signal: np.ndarray,
    beam_on_hint: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Extract qualifying ramp-down windows from one IC signal (one layer).

    Each window is background-subtracted and centred on the last beam-on
    slice (t=0).  Returns a list of raw (not normalised) windows.

    Parameters
    ----------
    beam_on_hint : optional bool array
        External beam-on mask (e.g. from a confidence column).  When
        provided, edge detection uses this instead of thresholding the
        signal itself — more reliable for noisy combined channels.
    """
    if np.isnan(signal).any():
        signal = signal.copy()
        signal[np.isnan(signal)] = np.nanmedian(signal)

    bg_array, bg_global, peak = _sliding_background(signal)
    if peak - bg_global < MIN_SIGNAL_SPAN:
        return []

    sig = signal - bg_array
    thresh = THRESHOLD_FRAC * (peak - bg_global)

    if beam_on_hint is not None:
        beam_on = beam_on_hint & (sig > thresh)
    else:
        beam_on = sig > thresh

    diff = np.diff(beam_on.astype(np.int8))
    edge_indices = np.where(diff == -1)[0] + 1

    windows: list[np.ndarray] = []
    n = len(sig)
    rise_tol = thresh * 0.5

    for idx in edge_indices:
        anchor = idx - 1
        start = anchor - PRE_OFF_SLICES
        end = anchor + POST_OFF_SLICES
        if start < 0 or end > n:
            continue
        if not np.all(beam_on[anchor - MIN_ON_SLICES + 1 : idx]):
            continue
        if np.any(beam_on[idx:end]):
            continue
        win = sig[start:end].copy()
        if beam_on_hint is not None:
            np.clip(win, 0.0, None, out=win)
        tail = win[PRE_OFF_SLICES + 1:]
        running_min = np.minimum.accumulate(tail)
        rise_mask = (tail - running_min) > rise_tol
        if rise_mask.any():
            cut = int(np.argmax(rise_mask))
            tail[cut:] = np.nan
        windows.append(win)

    return windows


def _normalise_windows(windows: list[np.ndarray]) -> np.ndarray | None:
    """Average and normalise a collection of ramp-down windows to 0-100 %."""
    if not windows:
        return None
    with np.errstate(all="ignore"):
        avg = np.nanmean(windows, axis=0)
    pk = np.nanmax(np.abs(avg))
    if not np.isfinite(pk) or pk < MIN_SIGNAL_SPAN:
        return None
    return (avg / pk) * 100.0


def _fit_decay(curve: np.ndarray, t_start_idx: int) -> dict | None:
    """Fit an exponential decay to a normalised ramp-down curve.

    Parameters
    ----------
    curve : array
        Full window (0-100 % normalised), length ``PRE_OFF_SLICES + POST_OFF_SLICES``.
    t_start_idx : int
        Index into *curve* where fitting begins (first fully-off sample).

    Returns a dict with keys ``model``, ``tau`` (or ``tau1``/``tau2``),
    ``f_3dB``, ``r_squared``, ``fit_t``, ``fit_y``.
    """
    from scipy.optimize import curve_fit

    y = curve[t_start_idx:]
    valid = np.isfinite(y)
    if valid.sum() < 3:
        return None
    t = np.arange(len(y), dtype=float) * MS_PER_SLICE
    y = y[valid]
    t = t[valid]

    def _single(t, A, tau, C):
        return A * np.exp(-t / tau) + C

    def _double(t, A1, tau1, A2, tau2, C):
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + C

    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot < 1e-12:
        return None

    # --- single exponential ---
    try:
        p0_s = [y[0], 1.0, 0.0]
        popt_s, _ = curve_fit(
            _single, t, y, p0=p0_s,
            bounds=([0, 0.01, -20], [200, 50, 20]),
            maxfev=5000,
        )
        res_s = y - _single(t, *popt_s)
        r2_s = 1.0 - np.sum(res_s ** 2) / ss_tot
    except (RuntimeError, ValueError):
        r2_s = -1.0
        popt_s = None

    if popt_s is not None and r2_s >= 0.95:
        tau = float(popt_s[1])
        fit_t_full = np.linspace(t[0], t[-1], 200)
        return {
            "model": "single",
            "tau": tau,
            "f_3dB": 1.0 / (2.0 * np.pi * tau * 1e-3),
            "r_squared": float(r2_s),
            "fit_t": fit_t_full + t_start_idx * MS_PER_SLICE,
            "fit_y": _single(fit_t_full, *popt_s),
        }

    # --- double exponential fallback ---
    try:
        p0_d = [y[0] * 0.7, 0.3, y[0] * 0.3, 2.0, 0.0]
        popt_d, _ = curve_fit(
            _double, t, y, p0=p0_d,
            bounds=([0, 0.01, 0, 0.01, -20], [200, 50, 200, 50, 20]),
            maxfev=10000,
        )
        res_d = y - _double(t, *popt_d)
        r2_d = 1.0 - np.sum(res_d ** 2) / ss_tot
    except (RuntimeError, ValueError):
        r2_d = -1.0
        popt_d = None

    if popt_d is not None and r2_d > (r2_s if popt_s is not None else -1):
        A1, tau1, A2, tau2, _C = popt_d
        if tau1 > tau2:
            A1, tau1, A2, tau2 = A2, tau2, A1, tau1
        fit_t_full = np.linspace(t[0], t[-1], 200)
        return {
            "model": "double",
            "tau1": float(tau1),
            "tau2": float(tau2),
            "A1": float(A1),
            "A2": float(A2),
            "f_3dB_fast": 1.0 / (2.0 * np.pi * float(tau1) * 1e-3),
            "f_3dB_slow": 1.0 / (2.0 * np.pi * float(tau2) * 1e-3),
            "r_squared": float(r2_d),
            "fit_t": fit_t_full + t_start_idx * MS_PER_SLICE,
            "fit_y": _double(fit_t_full, *popt_d),
        }

    if popt_s is not None:
        tau = float(popt_s[1])
        fit_t_full = np.linspace(t[0], t[-1], 200)
        return {
            "model": "single",
            "tau": tau,
            "f_3dB": 1.0 / (2.0 * np.pi * tau * 1e-3),
            "r_squared": float(r2_s),
            "fit_t": fit_t_full + t_start_idx * MS_PER_SLICE,
            "fit_y": _single(fit_t_full, *popt_s),
        }
    return None


def _format_decay_fit_label(fit: dict, *, prefix: str = "") -> str:
    """Compact decay-fit legend text: time constant(s) and R² only."""
    if fit["model"] == "single":
        tau_part = f"\u03C4={fit['tau']:.2f} ms"
    else:
        tau_part = f"\u03C4={fit['tau1']:.2f}/{fit['tau2']:.2f} ms"
    label = f"{tau_part}  R\u00B2={fit['r_squared']:.2f}"
    return f"{prefix}{label}" if prefix else label


def _extract_rampdown_curves(session_id: str, base_dir: str):
    """Extract averaged, normalised beam-off ramp-down curves per energy.

    Windows from all layers sharing the same energy are pooled before
    averaging, so single-energy sessions still produce good curves from
    the combined data of all layers.

    Returns
    -------
    dict mapping energy (float) -> dict with keys
        ``ic1_curve``, ``ic2_curve``, ``ic3_curve``
        each a 1-D array of length ``PRE_OFF_SLICES + POST_OFF_SLICES``
        scaled to 0–100 % (or None if that IC had no valid edges).
    Returns None on failure.
    """
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    input_map = load_session_csv(src, "input_map.csv")
    if input_map is None:
        return None

    lookups = build_energy_lookups(input_map)
    if lookups is None:
        return None
    energy_by_layer_id, energy_by_idx = lookups

    frames = load_session_timeslice_device_units(src)
    if not frames:
        return None

    signal_source = resolve_ic_rampdown_source(frames[0].columns)
    if signal_source is None:
        return None

    layer_col = resolve_col(frames[0].columns, C_LAYER_ID) or C_LAYER_ID

    windows_by_energy: dict[float, dict[str, list[np.ndarray]]] = {}

    for frame_idx, df in enumerate(frames):
        energy = resolve_frame_energy(
            df,
            frame_idx,
            energy_by_layer=energy_by_layer_id,
            energy_by_idx=energy_by_idx,
            layer_col=layer_col,
        )
        if energy is None:
            continue

        if energy not in windows_by_energy:
            windows_by_energy[energy] = {"ic1": [], "ic2": [], "ic3": []}
        bucket = windows_by_energy[energy]

        for ic_key in ("ic1", "ic2", "ic3"):
            sig = ic_rampdown_signal(df, signal_source, ic_key)
            if sig is not None:
                bucket[ic_key].extend(_extract_windows(sig))

    _IC_KEYS = ("ic1", "ic2", "ic3")

    per_energy: dict[float, dict[str, np.ndarray | None]] = {}
    all_windows: dict[str, list[np.ndarray]] = {k: [] for k in _IC_KEYS}

    for energy, bucket in windows_by_energy.items():
        curves = {f"{k}_curve": _normalise_windows(bucket[k]) for k in _IC_KEYS}
        if all(v is None for v in curves.values()):
            continue
        per_energy[energy] = curves
        for k in _IC_KEYS:
            all_windows[k].extend(bucket[k])

    if not per_energy:
        return None

    return {
        "signal_mode": signal_source.mode,
        "per_energy": per_energy,
        "avg": {f"{k}_curve": _normalise_windows(all_windows[k]) for k in _IC_KEYS},
    }


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Run beam-off ramp-down analysis and show matplotlib window."""
    del settings

    if not session_ids:
        print("No sessions selected")
        return

    session_results: dict[str, dict] = {}
    for sid in session_ids:
        result = _extract_rampdown_curves(sid, base_dir)
        if result is not None:
            session_results[sid] = result
        else:
            print(
                f"  {sid}: no ramp-down data "
                "(missing IC dose/current columns or no beam-off edges)",
            )

    if not session_results:
        print("No valid ramp-down data found for any session")
        return

    if not any(
        session_results[sid]["avg"].get(f"{ic}_curve") is not None
        for sid in session_results
        for ic in ("ic1", "ic2", "ic3")
    ):
        print("No valid ramp-down curves found for any session")
        return

    loaded_ids = list(session_results.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    use_dose = all(
        session_results[sid].get("signal_mode") == "dose" for sid in loaded_ids
    )
    ic_unit = "dDose/dt" if use_dose else "current"
    ic_defs_full = [
        ("ic1_curve", f"IC1 {ic_unit}"),
        ("ic2_curve", f"IC2 {ic_unit}"),
        ("ic3_curve", f"IC3 {ic_unit}"),
    ]
    ic_defs = [
        (key, label) for key, label in ic_defs_full
        if any(r["avg"].get(key) is not None for r in session_results.values())
    ]
    n_ic = len(ic_defs)

    t_axis = np.arange(-PRE_OFF_SLICES, POST_OFF_SLICES, dtype=float)

    heatmap_session_ids = [
        sid for sid in loaded_ids
        if len(session_results[sid]["per_energy"]) > 1
    ]
    n_heatmap_rows = len(heatmap_session_ids)

    n_rows = 1 + n_heatmap_rows  # row 0 = averages, rows 1..N = heatmaps
    height_ratios = [1] + [1.3] * n_heatmap_rows
    has_heatmap_cbar = n_heatmap_rows > 0
    n_cols_grid = n_ic + (1 if has_heatmap_cbar else 0)
    width_ratios = [1] * n_ic
    if has_heatmap_cbar:
        width_ratios.append(0.035)

    plot_cols = list(range(n_ic))
    cbar_col = n_ic if has_heatmap_cbar else None

    fig, axes = plt.subplots(
        n_rows, n_cols_grid,
        figsize=(6 * n_ic + 0.6, 4 + 4 * n_heatmap_rows),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios, "width_ratios": width_ratios},
    )

    for row_idx in range(n_rows):
        if cbar_col is not None:
            axes[row_idx, cbar_col].set_visible(False)

    # --- Row 0: grand-average per session ---
    for (key, label), col in zip(ic_defs, plot_cols):
        ax = axes[0, col]
        for si, sid in enumerate(loaded_ids):
            curve = session_results[sid]["avg"][key]
            if curve is None:
                continue
            ax.plot(
                t_axis, curve,
                color=colors[si], linewidth=1.5,
            )
        ax.axvline(x=0.5, **REFLINE_KW)
        ax.axhline(y=100, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH)
        ax.yaxis.set_major_formatter(FuncFormatter(_symlog_pct_tick))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(label)
        if col == 0:
            ax.set_ylabel(f"{ic_unit} (%)")
        ax.set_ylim(-15, 110)
        ax.grid(**GRID_KW)
        ax.xaxis.set_major_locator(MultipleLocator(1))

    # --- Fit decay and legend Row 0 ---
    fit_start = PRE_OFF_SLICES + 1  # index into curve; skip t=0 transition
    t_offset = -PRE_OFF_SLICES  # t_axis[0] value

    for (key, label), col in zip(ic_defs, plot_cols):
        ax = axes[0, col]
        fit_legend_entries: list[tuple[str, str]] = []

        for si, sid in enumerate(loaded_ids):
            curve = session_results[sid]["avg"][key]
            if curve is None:
                continue
            fit = _fit_decay(curve, fit_start)
            if fit is None:
                continue

            plot_t = fit["fit_t"] + t_offset
            line_color = trend_line_color(colors[si])
            ax.plot(
                plot_t, fit["fit_y"],
                color=line_color, alpha=0.85, **FIT_LINE_KW,
            )
            prefix = trend_session_prefix(sid, n_sessions=len(loaded_ids))
            fit_legend_entries.append(
                (_format_decay_fit_label(fit, prefix=prefix), line_color),
            )

        if fit_legend_entries:
            make_trend_legend(
                ax, fit_legend_entries, fontsize=7,
                line_kw=FIT_LINE_KW,
            )

    # --- Rows 1..N: one heatmap row per multi-energy session (ramp-down only: t >= 1) ---
    cbar_axes: list = []
    if n_heatmap_rows:
        heatmap_cmap = "turbo"
        ramp_start = PRE_OFF_SLICES + 1
        ramp_t = t_axis[ramp_start:]
        ramp_len = len(ramp_t)

        heatmap_vals: list[float] = []
        for sid in heatmap_session_ids:
            for pe_data in session_results[sid]["per_energy"].values():
                for key, _ in ic_defs:
                    c = pe_data[key]
                    if c is not None:
                        ramp = c[ramp_start:]
                        heatmap_vals.extend(ramp[np.isfinite(ramp)])
        heat_vmin = 0.0
        heat_vmax = float(np.max(heatmap_vals)) if heatmap_vals else 100.0

        for si, sid in enumerate(heatmap_session_ids):
            row = 1 + si
            pe = session_results[sid]["per_energy"]
            energies = sorted(pe.keys())

            _UPSAMPLE_X = 8
            x_orig = np.arange(ramp_len, dtype=float)
            x_fine = np.linspace(0, ramp_len - 1, ramp_len * _UPSAMPLE_X)

            for (key, label), col in zip(ic_defs, plot_cols):
                ax = axes[row, col]
                img = np.full((len(energies), ramp_len), np.nan)
                for ei, e in enumerate(energies):
                    c = pe[e][key]
                    if c is not None:
                        img[ei, :] = c[ramp_start:]

                img_smooth = np.full((len(energies), len(x_fine)), np.nan)
                for ri in range(len(energies)):
                    if np.any(np.isfinite(img[ri])):
                        img_smooth[ri] = np.interp(x_fine, x_orig, img[ri])

                y_lo, y_hi = _heatmap_energy_extent(energies)
                extent = [ramp_t[0] - 0.5, ramp_t[-1] + 0.5, y_lo, y_hi]
                ax.imshow(
                    img_smooth, aspect="auto", origin="lower",
                    extent=extent, cmap=heatmap_cmap,
                    vmin=heat_vmin, vmax=heat_vmax,
                    interpolation="nearest",
                )
                if col == 0:
                    ax.set_ylabel(f"{sid}\nEnergy (MeV)", fontsize=8)
                if row == n_rows - 1:
                    ax.set_xlabel("Time after beam-off (ms)")
                ax.xaxis.set_major_locator(MultipleLocator(1))

        if cbar_col is not None:
            cbar_ax = fig.add_subplot(
                axes[1, cbar_col].get_gridspec()[1:, cbar_col],
            )
            cbar_ax.set_visible(True)
            cbar = fig.colorbar(
                plt.cm.ScalarMappable(
                    cmap=heatmap_cmap,
                    norm=mcolors.Normalize(heat_vmin, heat_vmax),
                ),
                cax=cbar_ax,
            )
            cbar.set_label(f"{ic_unit} (%)")
            cbar_axes.append(cbar_ax)

    finish_view(
        fig,
        f"Beam-Off Ramp-Down Curves  ({ic_unit}, normalised to beam-on)",
        loaded_ids,
        colors,
        base_dir=base_dir,
        show=False,
    )
    for cbar_ax in cbar_axes:
        pos = cbar_ax.get_position()
        cbar_ax.set_position([pos.x0 - 0.006, pos.y0, pos.width, pos.height])
    plt.show()
