"""Current ratios vs energy from spot-peak currents (median / mean / filtered)."""

import logging

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from ..common import (
    C_ENERGY,
    C_IC1_CURRENT,
    C_IC2_CURRENT,
    C_IC3_CURRENT_A,
    C_IC3_CURRENT_B,
    C_IC3_CURRENT_C,
    C_IC3_CURRENT_D,
    C_LAYER_ID,
    resolve_concept_column,
    annotate_slopes,
    make_session_legend,
    DEFAULT_SESSION_COLORS,
    SUPTITLE_KW,
)
from ..common import subtract_background_frames
from ..common.session_source import (
    load_session_csv,
    load_session_timeslice_device_units,
    resolve_session_source,
)

_log = logging.getLogger(__name__)

_NOISE_FLOOR_NA = 5.0  # minimum current (nA) to be considered part of a pulse

_IC_CURRENT_COLS = {
    "ic1": [C_IC1_CURRENT],
    "ic2": [C_IC2_CURRENT],
    "ic3": [C_IC3_CURRENT_A, C_IC3_CURRENT_B, C_IC3_CURRENT_C, C_IC3_CURRENT_D],
}


def _extract_spot_peaks(signal: np.ndarray) -> np.ndarray:
    """Find peak current of each individual spot pulse in *signal*."""
    above = signal > _NOISE_FLOOR_NA
    if not above.any():
        return np.array([])
    edges = np.diff(above.astype(np.int8))
    starts = np.where(edges == 1)[0] + 1
    stops = np.where(edges == -1)[0] + 1
    if above[0]:
        starts = np.concatenate([[0], starts])
    if above[-1]:
        stops = np.concatenate([stops, [len(signal)]])
    peaks = []
    for s, e in zip(starts, stops):
        if e - s >= 2:
            peaks.append(signal[s:e].max())
    return np.array(peaks, dtype=np.float64)


def _load_current_ratios(session_id: str, base_dir: str, *, bg_subtract: bool = False) -> dict | None:
    """Load timeslice data, extract spot peaks per IC/energy, compute median ratios."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    input_map = load_session_csv(src, "input_map.csv")
    if input_map is None:
        return None

    col_energy = resolve_concept_column(input_map.columns, C_ENERGY)
    col_layer_im = resolve_concept_column(input_map.columns, C_LAYER_ID)
    if col_energy is None:
        return None

    frames = load_session_timeslice_device_units(src)
    if not frames:
        return None
    if bg_subtract:
        subtract_background_frames(frames)

    df0 = frames[0]
    has_ic1 = any(c in df0.columns for c in _IC_CURRENT_COLS["ic1"])
    has_ic2 = any(c in df0.columns for c in _IC_CURRENT_COLS["ic2"])
    has_ic3 = all(c in df0.columns for c in _IC_CURRENT_COLS["ic3"])
    if not (has_ic1 and has_ic2):
        return None

    ic_keys = ["ic1", "ic2"]
    if has_ic3:
        ic_keys.append("ic3")

    ordered_energies = list(dict.fromkeys(input_map[col_energy].values))
    energy_by_idx = {i: float(e) for i, e in enumerate(ordered_energies)}

    energy_by_layer: dict | None = None
    if col_layer_im is not None:
        unique_layers = input_map[col_layer_im].nunique()
        if unique_layers > 1:
            energy_by_layer = (
                input_map.groupby(col_layer_im)[col_energy].first().to_dict()
            )

    energies_out: list[float] = []
    ic_spot_peaks: dict[str, list[np.ndarray]] = {ic: [] for ic in ic_keys}

    for frame_i, df in enumerate(frames):
        energy = None
        if energy_by_layer is not None and C_LAYER_ID in df.columns:
            lid = df[C_LAYER_ID].iloc[0]
            energy = energy_by_layer.get(lid)
        if energy is None and "_layer_idx" in df.columns:
            idx = int(df["_layer_idx"].iloc[0])
            energy = energy_by_idx.get(idx)
        if energy is None:
            energy = energy_by_idx.get(frame_i, 0.0)

        energies_out.append(energy)

        for ic in ic_keys:
            cur_cols = [c for c in _IC_CURRENT_COLS[ic] if c in df.columns]
            if not cur_cols:
                ic_spot_peaks[ic].append(np.array([]))
                continue
            row_sums = df[cur_cols].sum(axis=1).to_numpy(dtype=np.float64, na_value=0.0)
            row_sums[~np.isfinite(row_sums)] = 0.0
            ic_spot_peaks[ic].append(_extract_spot_peaks(row_sums))

    if not energies_out:
        return None

    n_layers = len(energies_out)
    _MIN_SPOTS = 3
    energy_arr = np.array(energies_out, dtype=float)

    def _per_layer(ic, agg):
        """Per-layer point estimate (median or mean)."""
        out = np.full(n_layers, np.nan)
        for i in range(n_layers):
            pk = ic_spot_peaks[ic][i]
            if len(pk) < _MIN_SPOTS:
                continue
            out[i] = float(agg(pk))
        return out

    def _filtfilt_curve(ic, cutoff=0.08, order=2):
        """Zero-phase Butterworth low-pass on per-energy medians.

        Data is sorted by energy before filtering and mapped back so
        the result aligns with the original layer order.
        """
        medians = _per_layer(ic, np.median)
        ok = np.isfinite(energy_arr) & np.isfinite(medians)
        if ok.sum() < 2 * (order + 1):
            return medians
        sort_idx = np.argsort(energy_arr[ok])
        vals_sorted = medians[ok][sort_idx]
        b, a = butter(order, cutoff, btype="low")
        padlen = min(3 * max(len(a), len(b)), len(vals_sorted) - 1)
        filtered = filtfilt(b, a, vals_sorted, padlen=padlen)
        smoothed = np.full(n_layers, np.nan)
        out_slots = np.where(ok)[0]
        smoothed[out_slots[sort_idx]] = filtered
        return smoothed

    result: dict = {"energy": pd.Series(energies_out, dtype=float)}
    for ic in ic_keys:
        result[f"{ic}_spot_peaks"] = ic_spot_peaks[ic]

    estimators = {
        "median": lambda ic: _per_layer(ic, np.median),
        "mean":   lambda ic: _per_layer(ic, np.mean),
        "filt":   lambda ic: _filtfilt_curve(ic),
    }

    for method, est_fn in estimators.items():
        v1 = est_fn("ic1")
        v2 = est_fn("ic2")
        result[f"ic1_{method}"] = v1
        result[f"ic2_{method}"] = v2
        with np.errstate(divide="ignore", invalid="ignore"):
            result[f"ic21_{method}"] = (v2 - v1) / ((v2 + v1) / 2.0) * 100.0
            if "ic3" in ic_keys:
                v3 = est_fn("ic3")
                result[f"ic3_{method}"] = v3
                result[f"ic31_{method}"] = (v3 - v1) / ((v3 + v1) / 2.0) * 100.0
                result[f"ic32_{method}"] = (v3 - v2) / ((v3 + v2) / 2.0) * 100.0

    return result


def _trend_line_color(face_color):
    try:
        rgb = mcolors.to_rgb(face_color)
    except ValueError:
        rgb = mcolors.to_rgb("C0")
    return tuple(max(0.0, c * 0.55) for c in rgb)


_METHODS = [
    ("filt",   "Filtered"),
    ("median", "Median"),
    ("mean",   "Mean"),
]

_RATIO_PAIRS = [
    ("ic21", "IC2 / IC1", "ic1", "ic2"),
    ("ic31", "IC3 / IC1", "ic1", "ic3"),
    ("ic32", "IC3 / IC2", "ic2", "ic3"),
]

_HEATMAP_ICS = ["ic1", "ic2", "ic3"]
_HEATMAP_NBINS_Y = 50


def _plot_current_heatmap(ax, session_data, ic_name: str, loaded_ids, sess_colors):
    """2D heatmap of per-spot peak current vs energy for one IC, all sessions.

    X = energy (MeV), Y = peak current (nA), colour = probability density
    normalised per energy column.  A polynomial fit line is drawn per session.
    """
    peaks_key = f"{ic_name}_spot_peaks"

    combined: dict[float, list[float]] = {}
    per_session: dict[str, dict[float, list[float]]] = {}

    for sid in loaded_ids:
        data = session_data.get(sid)
        if data is None:
            continue
        energies = np.asarray(data["energy"], dtype=float)
        peaks_list = data.get(peaks_key, [])
        sess_dict: dict[float, list[float]] = {}
        for i, e in enumerate(energies):
            if not np.isfinite(e):
                continue
            if i < len(peaks_list) and len(peaks_list[i]) > 0:
                combined.setdefault(e, []).extend(peaks_list[i])
                sess_dict.setdefault(e, []).extend(peaks_list[i])
        per_session[sid] = sess_dict

    if not combined:
        ax.set_ylabel("Peak Current (nA)")
        ax.grid(visible=True, alpha=0.3)
        return

    unique_e = np.sort(np.array(list(combined.keys())))
    if len(unique_e) < 2:
        ax.set_ylabel("Peak Current (nA)")
        return

    all_peaks = np.concatenate(list(combined.values()))
    y_lo = max(0, np.percentile(all_peaks, 0.5) * 0.8)
    y_hi = np.percentile(all_peaks, 99.5) * 1.15
    y_edges = np.linspace(y_lo, y_hi, _HEATMAP_NBINS_Y + 1)
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    density = np.zeros((len(unique_e), _HEATMAP_NBINS_Y))
    for col_i, e in enumerate(unique_e):
        vals = np.array(combined[e])
        h, _ = np.histogram(vals, bins=y_edges)
        total = h.sum()
        if total > 0:
            density[col_i] = h / total

    im = ax.pcolormesh(
        unique_e, y_centers, density.T,
        cmap="turbo", shading="nearest", zorder=1,
    )
    ax.figure.colorbar(im, ax=ax, pad=0.02, aspect=30, label="P(I)")

    for si, sid in enumerate(loaded_ids):
        sess_dict = per_session.get(sid, {})
        me, mv = [], []
        for e in sorted(sess_dict):
            vals = sess_dict[e]
            if len(vals) >= 3:
                me.append(e)
                mv.append(float(np.median(vals)))
        if len(me) >= 6:
            mv_a = np.array(mv)
            b, a = butter(2, 0.08, btype="low")
            padlen = min(3 * max(len(a), len(b)), len(mv_a) - 1)
            mv_filt = filtfilt(b, a, mv_a, padlen=padlen)
            ax.plot(me, mv_filt,
                    color=sess_colors[si], linewidth=1.5, linestyle="--",
                    alpha=0.7, zorder=4, label=sid)

    ax.set_ylabel(f"{ic_name.upper()} Peak (nA)")
    ax.set_facecolor("black")
    ax.legend(fontsize=7, loc="upper left")


def _plot_ratio_vs_energy(ax, session_data, ratio_key, loaded_ids, colors):
    """Scatter + trend line of one ratio value per energy layer."""
    n_sessions = len(loaded_ids)
    slope_labels: list[tuple[str, tuple]] = []

    for si, sid in enumerate(loaded_ids):
        data = session_data[sid]
        if ratio_key not in data:
            continue
        e = np.asarray(data["energy"], dtype=float)
        r = np.asarray(data[ratio_key], dtype=float)
        ok = np.isfinite(e) & np.isfinite(r)
        e, r = e[ok], r[ok]
        if e.size == 0:
            continue

        ax.scatter(e, r, c=colors[si], s=18, alpha=0.85, edgecolors="none", zorder=3)

        if e.size >= 2:
            keep = np.ones(e.size, dtype=bool)
            for _ in range(3):
                if keep.sum() < 3:
                    break
                slope, intercept = np.polyfit(e[keep], r[keep], 1)
                resid = r - (slope * e + intercept)
                med = np.median(resid[keep])
                sigma = np.median(np.abs(resid[keep] - med)) * 1.4826
                if sigma < 1e-12:
                    break
                keep = np.abs(resid - med) <= 2.0 * sigma

            slope, intercept = np.polyfit(e[keep], r[keep], 1)

            rejected = ~keep
            if rejected.any():
                ax.scatter(e[rejected], r[rejected], c=colors[si], s=18,
                           alpha=0.25, edgecolors="red", linewidths=0.8,
                           zorder=2)

            e_range = np.array([e.min(), e.max()])
            line_color = _trend_line_color(colors[si])
            ax.plot(e_range, slope * e_range + intercept, color=line_color,
                    linewidth=2.0, linestyle="-", zorder=4)
            fit_delta = slope * (e.max() - e.min())
            prefix = f"{sid}: " if n_sessions > 1 else ""
            slope_labels.append(
                (f"{prefix}{slope:+.4g} %/MeV  (Δ {fit_delta:+.3g}%)", line_color)
            )

    if slope_labels:
        annotate_slopes(ax, slope_labels)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.grid(visible=True, alpha=0.3)


_IC_MARKERS = {"ic1": "o", "ic2": "s", "ic3": "D"}


def _plot_filtered_pair(ax, session_data, ic_a, ic_b, method, loaded_ids, sess_colors):
    """Plot the two filtered IC curves that form a ratio, instead of the ratio itself."""
    n_sessions = len(loaded_ids)

    for si, sid in enumerate(loaded_ids):
        data = session_data.get(sid)
        if data is None:
            continue
        e = np.asarray(data["energy"], dtype=float)

        for ic_name in (ic_a, ic_b):
            key = f"{ic_name}_{method}"
            if key not in data:
                continue
            vals = np.asarray(data[key], dtype=float)
            ok = np.isfinite(e) & np.isfinite(vals)
            if not ok.any():
                continue
            marker = _IC_MARKERS.get(ic_name, "o")
            label = f"{ic_name.upper()} ({sid})" if n_sessions > 1 else ic_name.upper()
            ax.scatter(e[ok], vals[ok], c=sess_colors[si], s=14, alpha=0.7,
                       marker=marker, edgecolors="none", zorder=3, label=label)

    ax.set_ylabel("Filtered Peak (nA)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(visible=True, alpha=0.3)


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Run current ratios analysis and show matplotlib window."""
    if not session_ids:
        _log.debug("No sessions selected")
        return

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        bg = settings.bg_subtract if settings else False
        data = _load_current_ratios(sid, base_dir, bg_subtract=bg)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        _log.debug("No valid timeslice data found for any session")
        return

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[:len(loaded_ids)]

    has_ic3 = any("ic31_median" in d for d in session_data.values())
    n_ratio_rows = 3 if has_ic3 else 1
    n_heatmap_rows = 3 if has_ic3 else 2
    n_rows = max(n_ratio_rows, n_heatmap_rows)
    n_methods = len(_METHODS)
    n_cols = 1 + n_methods  # heatmap + method columns

    # Column order: heatmap, filtered, median, mean
    col_order = [("heatmap", "Peak Distribution", 2)]
    for mi, (mk, ml) in enumerate(_METHODS):
        col_order.append((mk, ml, 3))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows + 1),
        squeeze=False,
        width_ratios=[w for _, _, w in col_order],
    )
    fig.suptitle(
        "Current Ratios vs Energy  (heatmap · filtered · median · mean)",
        **SUPTITLE_KW,
    )

    session_data_g3 = {k: v for k, v in session_data.items() if "ic31_median" in v}
    colors_g3 = [colors[loaded_ids.index(sid)] for sid in session_data_g3]

    row_data = [
        (session_data, loaded_ids, colors),
        (session_data_g3, list(session_data_g3.keys()), colors_g3),
        (session_data_g3, list(session_data_g3.keys()), colors_g3),
    ]

    pct_axes: list = []

    for row in range(n_rows):
        has_ratio = row < n_ratio_rows
        if has_ratio:
            pair_key, pair_label, ic_a, ic_b = _RATIO_PAIRS[row]
            sdata, sids, scols = row_data[row]
        else:
            sdata, sids, scols = session_data, loaded_ids, colors

        for ci, (col_key, col_label, _) in enumerate(col_order):
            ax = axes[row, ci]

            if col_key == "heatmap":
                _plot_current_heatmap(ax, sdata, _HEATMAP_ICS[row], sids, scols)
            elif col_key == "filt":
                if has_ratio and sdata:
                    _plot_filtered_pair(ax, sdata, ic_a, ic_b, col_key, sids, scols)
                    ax.set_ylabel(f"{pair_label}  filtered (nA)")
                else:
                    ax.set_visible(False)
            else:
                if has_ratio and sdata:
                    ratio_key = f"{pair_key}_{col_key}"
                    _plot_ratio_vs_energy(ax, sdata, ratio_key, sids, scols)
                    ax.set_ylabel(f"{pair_label}  Δ% ({col_label})")
                    pct_axes.append(ax)
                else:
                    ax.set_visible(False)

            if row == 0:
                ax.set_title(col_label, fontsize=10)
            if row == n_rows - 1 and ax.get_visible():
                ax.set_xlabel("Energy (MeV)")

    if pct_axes:
        y_lo = min(ax.get_ylim()[0] for ax in pct_axes)
        y_hi = max(ax.get_ylim()[1] for ax in pct_axes)
        for ax in pct_axes:
            ax.set_ylim(y_lo, y_hi)

    # Put session legend on the first ratio column
    legend_ax = next((ax for ax in pct_axes if ax.get_visible()), axes[0, 0])
    make_session_legend(legend_ax, loaded_ids, colors)

    plt.tight_layout()
    plt.show()
