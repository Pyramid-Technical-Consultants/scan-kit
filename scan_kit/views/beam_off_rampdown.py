"""Beam-off ramp-down curve analysis (IC1, IC2, IC3) from timeslice data."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

from ..common import (
    C_ENERGY,
    C_IC1_CURRENT,
    C_IC2_CURRENT,
    C_IC1_STRIP_SUM,
    C_IC2_STRIP_SUM,
    C_IC3_CURRENT_A,
    C_IC3_CURRENT_B,
    C_IC3_CURRENT_C,
    C_IC3_CURRENT_D,
    C_LAYER_ID,
    DEFAULT_SESSION_COLORS,
    finish_view,
    GRID_KW,
    REFLINE_KW,
)
from ..common import sliding_background
from ..common.session_source import (
    load_session_csv,
    load_session_timeslice_device_units,
    resolve_session_source,
)

import logging

_log = logging.getLogger(__name__)

# ---- Tweakable window parameters ------------------------------------------
PRE_OFF_SLICES = 2  # timeslices shown before the falling edge
POST_OFF_SLICES = 9  # timeslices shown after the falling edge
MIN_ON_SLICES = 2  # minimum consecutive above-threshold slices right
# before the falling edge (filters brief spikes)
THRESHOLD_FRAC = 0.10  # fraction of (peak − background) used to define the
# beam-on / beam-off boundary on IC1 current
# ---------------------------------------------------------------------------

_CONFIDENCE_COLS = {
    "ic1": ("r_ic1_x_confidence", "r_ic1_y_confidence"),
    "ic2": ("r_ic2_x_confidence", "r_ic2_y_confidence"),
    "ic3": ("r_px3_1_confidence",),
}
_CONFIDENCE_THRESHOLD = 0.0

_TIMESLICE_COLS = [
    C_LAYER_ID,
    C_IC1_CURRENT,
    C_IC2_CURRENT,
    C_IC1_STRIP_SUM,
    C_IC2_STRIP_SUM,
    C_IC3_CURRENT_A,
    C_IC3_CURRENT_B,
    C_IC3_CURRENT_C,
    C_IC3_CURRENT_D,
    *[c for cols in _CONFIDENCE_COLS.values() for c in cols],
]


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
    if peak - bg_global < 1.0:
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
        signal itself — much more reliable for noisy channels like
        strip sums.
    """
    if np.isnan(signal).any():
        signal = signal.copy()
        signal[np.isnan(signal)] = np.nanmedian(signal)

    bg_array, bg_global, peak = _sliding_background(signal)
    if peak - bg_global < 1.0:
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
    if pk < 1.0:
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


MS_PER_SLICE = 1.0


def _fit_per_energy_taus(
    per_energy: dict[float, dict],
    ic_key: str,
    t_start_idx: int,
) -> list[float]:
    """Collect dominant tau values across all energies for one IC.

    For single-exp fits returns tau; for double-exp returns the
    amplitude-weighted effective tau.
    """
    taus: list[float] = []
    for energy in sorted(per_energy):
        curve = per_energy[energy].get(ic_key)
        if curve is None:
            continue
        result = _fit_decay(curve, t_start_idx)
        if result is None:
            continue
        if result["model"] == "single":
            taus.append(result["tau"])
        else:
            A1, tau1 = result["A1"], result["tau1"]
            A2, tau2 = result["A2"], result["tau2"]
            taus.append((A1 * tau1 + A2 * tau2) / (A1 + A2))
    return taus


def _confidence_beam_on(df, ic_key: str) -> np.ndarray | None:
    """Return a boolean beam-on mask by OR-ing all confidence axes for an IC."""
    cols = _CONFIDENCE_COLS.get(ic_key, ())
    present = [c for c in cols if c in df.columns]
    if not present:
        return None
    mask = df[present[0]].values.astype(float) > _CONFIDENCE_THRESHOLD
    for c in present[1:]:
        mask = mask | (df[c].values.astype(float) > _CONFIDENCE_THRESHOLD)
    return mask


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

    if C_ENERGY not in input_map.columns:
        return None

    energy_by_layer_id: dict | None = None
    energy_by_idx: dict[int, float] | None = None

    if C_LAYER_ID in input_map.columns:
        energy_by_layer_id = input_map.groupby(C_LAYER_ID)[C_ENERGY].first().to_dict()

    if energy_by_layer_id is None or len(energy_by_layer_id) <= 1:
        ordered_energies = list(dict.fromkeys(input_map[C_ENERGY].values))
        energy_by_idx = {i: e for i, e in enumerate(ordered_energies)}

    frames = load_session_timeslice_device_units(src, usecols=_TIMESLICE_COLS)
    if not frames:
        return None

    ic3_cols = [C_IC3_CURRENT_A, C_IC3_CURRENT_B, C_IC3_CURRENT_C, C_IC3_CURRENT_D]

    windows_by_energy: dict[float, dict[str, list[np.ndarray]]] = {}

    for df in frames:
        energy = None
        if energy_by_idx is not None and "_layer_idx" in df.columns:
            idx = int(df["_layer_idx"].iloc[0])
            energy = energy_by_idx.get(idx)
        if energy is None and energy_by_layer_id is not None and C_LAYER_ID in df.columns:
            layer_id = df[C_LAYER_ID].iloc[0]
            energy = energy_by_layer_id.get(layer_id)
        if energy is None:
            continue

        if C_IC1_CURRENT not in df.columns and C_IC2_CURRENT not in df.columns:
            continue

        if energy not in windows_by_energy:
            windows_by_energy[energy] = {
                "ic1": [], "ic2": [], "ic1_ss": [], "ic2_ss": [], "ic3": [],
            }
        bucket = windows_by_energy[energy]

        ic1_hint = _confidence_beam_on(df, "ic1")
        ic2_hint = _confidence_beam_on(df, "ic2")

        if C_IC1_CURRENT in df.columns:
            bucket["ic1"].extend(_extract_windows(df[C_IC1_CURRENT].values))
        if C_IC2_CURRENT in df.columns:
            bucket["ic2"].extend(_extract_windows(df[C_IC2_CURRENT].values))
        if C_IC1_STRIP_SUM in df.columns:
            bucket["ic1_ss"].extend(
                _extract_windows(df[C_IC1_STRIP_SUM].values, beam_on_hint=ic1_hint)
            )
        if C_IC2_STRIP_SUM in df.columns:
            bucket["ic2_ss"].extend(
                _extract_windows(df[C_IC2_STRIP_SUM].values, beam_on_hint=ic2_hint)
            )

        has_ic3 = all(col in df.columns for col in ic3_cols)
        if has_ic3:
            ic3_sig = (
                df[C_IC3_CURRENT_A].values
                + df[C_IC3_CURRENT_B].values
                + df[C_IC3_CURRENT_C].values
                + df[C_IC3_CURRENT_D].values
            )
            ic3_hint = _confidence_beam_on(df, "ic3")
            bucket["ic3"].extend(_extract_windows(ic3_sig, beam_on_hint=ic3_hint))

    _IC_KEYS = ("ic1", "ic2", "ic1_ss", "ic2_ss", "ic3")

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
        "per_energy": per_energy,
        "avg": {f"{k}_curve": _normalise_windows(all_windows[k]) for k in _IC_KEYS},
    }


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Run beam-off ramp-down analysis and show matplotlib window."""
    if not session_ids:
        _log.debug("No sessions selected")
        return

    session_results: dict[str, dict] = {}
    for sid in session_ids:
        result = _extract_rampdown_curves(sid, base_dir)
        if result is not None:
            session_results[sid] = result

    if not session_results:
        _log.debug("No valid ramp-down data found for any session")
        return

    loaded_ids = list(session_results.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    ic_defs_full = [
        ("ic1_curve", "IC1"),
        ("ic2_curve", "IC2"),
        ("ic3_curve", "IC3 (A+B+C+D)"),
        ("ic1_ss_curve", "IC1 Strip Sum"),
        ("ic2_ss_curve", "IC2 Strip Sum"),
    ]
    ic_defs = [
        (key, label) for key, label in ic_defs_full
        if any(r["avg"].get(key) is not None for r in session_results.values())
    ]
    ic_defs_regular = [
        (key, label) for (key, label) in ic_defs if not key.endswith("_ss_curve")
    ]
    ic_defs_strip = [
        (key, label) for (key, label) in ic_defs if key.endswith("_ss_curve")
    ]
    ic_defs = ic_defs_regular + ic_defs_strip
    n_ic = len(ic_defs)

    t_axis = np.arange(-PRE_OFF_SLICES, POST_OFF_SLICES, dtype=float)

    all_energies: set[float] = set()
    for r in session_results.values():
        all_energies.update(r["per_energy"].keys())
    

    n_sessions = len(loaded_ids)
    n_rows = 1 + n_sessions  # row 0 = averages, rows 1..N = heatmaps
    height_ratios = [1] + [1.3] * n_sessions
    has_middle_regular_cbar = bool(ic_defs_regular and ic_defs_strip)
    has_strip_cbar = bool(ic_defs_strip)
    n_color_cols = int(has_middle_regular_cbar) + int(has_strip_cbar)
    n_cols_grid = n_ic + n_color_cols
    width_ratios = [1] * len(ic_defs_regular)
    if has_middle_regular_cbar:
        width_ratios.append(0.035)
    width_ratios += [1] * len(ic_defs_strip)
    if has_strip_cbar:
        width_ratios.append(0.035)

    plot_cols = list(range(len(ic_defs_regular)))
    strip_start = len(ic_defs_regular) + (1 if has_middle_regular_cbar else 0)
    plot_cols += list(range(strip_start, strip_start + len(ic_defs_strip)))
    middle_cbar_col = len(ic_defs_regular) if has_middle_regular_cbar else None
    strip_cbar_col = (n_cols_grid - 1) if has_strip_cbar else None

    fig, axes = plt.subplots(
        n_rows, n_cols_grid,
        figsize=(6 * n_ic + 0.6, 4 + 4 * n_sessions),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios, "width_ratios": width_ratios},
    )

    for row_idx in range(n_rows):
        if middle_cbar_col is not None:
            axes[row_idx, middle_cbar_col].set_visible(False)
        if strip_cbar_col is not None:
            axes[row_idx, strip_cbar_col].set_visible(False)

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
                label=sid,
            )
        ax.axvline(x=0.5, **REFLINE_KW)
        ax.axhline(y=100, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_title(label)
        if col == 0:
            ax.set_ylabel("Current (%)")
        ax.set_ylim(-15, 115)
        ax.grid(**GRID_KW)
        ax.xaxis.set_major_locator(MultipleLocator(1))

    # --- Fit decay and annotate Row 0 ---
    fit_start = PRE_OFF_SLICES + 1  # index into curve; skip t=0 transition
    t_offset = -PRE_OFF_SLICES  # t_axis[0] value

    for (key, label), col in zip(ic_defs, plot_cols):
        ax = axes[0, col]
        annotations: list[str] = []

        for si, sid in enumerate(loaded_ids):
            curve = session_results[sid]["avg"][key]
            if curve is None:
                continue
            fit = _fit_decay(curve, fit_start)
            if fit is None:
                continue

            plot_t = fit["fit_t"] + t_offset
            ax.plot(
                plot_t, fit["fit_y"],
                color=colors[si], linewidth=1.2, linestyle="--", alpha=0.85,
            )

            if fit["model"] == "single":
                tau_str = f"\u03C4 = {fit['tau']:.2f} ms"
                bw_str = f"f_3dB = {fit['f_3dB']:.0f} Hz"
            else:
                tau_str = (
                    f"\u03C4\u2081 = {fit['tau1']:.2f} ms  "
                    f"\u03C4\u2082 = {fit['tau2']:.2f} ms"
                )
                bw_str = (
                    f"f_3dB = {fit['f_3dB_fast']:.0f} / "
                    f"{fit['f_3dB_slow']:.0f} Hz"
                )

            pe_taus = _fit_per_energy_taus(
                session_results[sid]["per_energy"], key, fit_start,
            )
            if pe_taus:
                spread = (
                    f"Per-energy \u03C4: {np.mean(pe_taus):.2f} "
                    f"\u00B1 {np.std(pe_taus):.2f} ms  "
                    f"(n={len(pe_taus)})"
                )
            else:
                spread = ""

            prefix = f"{sid}: " if len(loaded_ids) > 1 else ""
            line = f"{prefix}{tau_str}   {bw_str}   R\u00B2={fit['r_squared']:.3f}"
            if spread:
                line += f"\n{' ' * len(prefix)}{spread}"
            annotations.append(line)

        if annotations:
            ax.text(
                0.97, 0.55, "\n".join(annotations),
                transform=ax.transAxes, fontsize=7,
                va="top", ha="right",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
            )

    axes[0, 0].legend(loc="upper right", fontsize=8)

    # --- Rows 1..N: one heatmap row per session (ramp-down only: t >= 1) ---
    heatmap_cmap = "turbo"
    ramp_start = PRE_OFF_SLICES + 1
    ramp_t = t_axis[ramp_start:]
    ramp_len = len(ramp_t)

    regular_keys = {k for k, _ in ic_defs_regular}
    strip_keys = {k for k, _ in ic_defs_strip}

    heatmap_vals_regular: list[float] = []
    heatmap_vals_strip: list[float] = []
    for r in session_results.values():
        for pe_data in r["per_energy"].values():
            for key, _ in ic_defs:
                c = pe_data[key]
                if c is not None:
                    ramp = c[ramp_start:]
                    vals = ramp[np.isfinite(ramp)]
                    if key in strip_keys:
                        heatmap_vals_strip.extend(vals)
                    else:
                        heatmap_vals_regular.extend(vals)
    heat_vmin_regular = 0.0
    heat_vmax_regular = (
        float(np.max(heatmap_vals_regular))
        if heatmap_vals_regular else 100.0
    )
    heat_vmin_strip = 0.0
    heat_vmax_strip = (
        float(np.max(heatmap_vals_strip))
        if heatmap_vals_strip else 100.0
    )

    for si, sid in enumerate(loaded_ids):
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

            extent = [ramp_t[0] - 0.5, ramp_t[-1] + 0.5,
                      energies[0], energies[-1]]
            ax.imshow(
                img_smooth, aspect="auto", origin="lower",
                extent=extent, cmap=heatmap_cmap,
                vmin=(heat_vmin_strip if key in strip_keys else heat_vmin_regular),
                vmax=(heat_vmax_strip if key in strip_keys else heat_vmax_regular),
                interpolation="nearest",
            )
            if col == 0:
                ax.set_ylabel(f"{sid}\nEnergy (MeV)", fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("Time after beam-off (ms)")
            ax.xaxis.set_major_locator(MultipleLocator(1))

    cbar_axes: list = []

    if regular_keys and middle_cbar_col is not None:
        cbar_ax_regular = fig.add_subplot(
            axes[1, middle_cbar_col].get_gridspec()[1:, middle_cbar_col],
        )
        cbar_ax_regular.set_visible(True)
        cbar_regular = fig.colorbar(
            plt.cm.ScalarMappable(
                cmap=heatmap_cmap,
                norm=mcolors.Normalize(heat_vmin_regular, heat_vmax_regular),
            ),
            cax=cbar_ax_regular,
        )
        cbar_regular.set_label("Current (%)")
        cbar_axes.append(cbar_ax_regular)

    if strip_keys and strip_cbar_col is not None:
        cbar_ax_strip = fig.add_subplot(
            axes[1, strip_cbar_col].get_gridspec()[1:, strip_cbar_col],
        )
        cbar_ax_strip.set_visible(True)
        cbar_strip = fig.colorbar(
            plt.cm.ScalarMappable(
                cmap=heatmap_cmap,
                norm=mcolors.Normalize(heat_vmin_strip, heat_vmax_strip),
            ),
            cax=cbar_ax_strip,
        )
        cbar_strip.set_label("Strip Current (%)")
        cbar_axes.append(cbar_ax_strip)

    finish_view(
        fig,
        "Beam-Off Ramp-Down Curves  (normalised to beam-on)",
        loaded_ids,
        colors,
        base_dir=base_dir,
    )
    for cbar_ax in cbar_axes:
        pos = cbar_ax.get_position()
        cbar_ax.set_position([pos.x0 - 0.006, pos.y0, pos.width, pos.height])
    plt.show()
