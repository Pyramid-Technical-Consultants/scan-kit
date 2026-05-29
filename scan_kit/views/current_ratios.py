"""Current ratios vs energy from beam-on mean IC currents."""

import logging

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
    add_scatter_trend,
    annotate_slopes,
    make_session_legend,
    DEFAULT_SESSION_COLORS,
    SUPTITLE_KW,
    apply_tight_layout,
)
from ..common import subtract_background_frames
from ..common.processing import _detect_beam_off_mask
from ..common.session_source import (
    load_session_csv,
    load_session_timeslice_device_units,
    resolve_session_source,
)

_log = logging.getLogger(__name__)

# ── Tunable parameters ──────────────────────────────────────────────────
LOWPASS_ORDER = 2  # Butterworth filter order
LOWPASS_CUTOFF = 0.07  # cutoff as fraction of Nyquist
OUTLIER_SIGMA = 2.0  # MAD-sigma multiplier for robust fit rejection
OUTLIER_ITERATIONS = 3  # robust fit re-weighting passes
HEATMAP_BINS = 100  # Y-axis histogram bins for beam-on current heatmap
DISPLAY_P95_FRAC = 0.75  # per-IC floor = p95 * frac, for heatmap/curve display only
MIN_BEAM_SAMPLES = 10  # min beam-on samples per layer for a valid estimate
# ─────────────────────────────────────────────────────────────────────────

_IC_CURRENT_COLS = {
    "ic1": [C_IC1_CURRENT],
    "ic2": [C_IC2_CURRENT],
    "ic3": [C_IC3_CURRENT_A, C_IC3_CURRENT_B, C_IC3_CURRENT_C, C_IC3_CURRENT_D],
}


def _lowpass(values: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """Butterworth zero-phase low-pass on values sorted by energy.

    Returns an array aligned with the original layer order.
    """
    ok = np.isfinite(energies) & np.isfinite(values)
    min_ok = 2 * (LOWPASS_ORDER + 1)
    if ok.sum() < min_ok:
        return values.copy()
    sort_idx = np.argsort(energies[ok])
    vals_sorted = values[ok][sort_idx]
    b, a = butter(LOWPASS_ORDER, LOWPASS_CUTOFF, btype="low")
    padlen = min(3 * max(len(a), len(b)), len(vals_sorted) - 1)
    smoothed_sorted = filtfilt(b, a, vals_sorted, padlen=padlen)
    out = np.full_like(values, np.nan)
    out[np.where(ok)[0][sort_idx]] = smoothed_sorted
    return out


def _sym_pct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Symmetric relative difference in percent."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / ((a + b) / 2.0) * 100.0


def _load_current_ratios(
    session_id: str, base_dir: str, *, bg_subtract: bool = False
) -> dict | None:
    """Load timeslice data, compute beam-on mean current per IC/energy."""
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
        if unique_layers >= 2:
            energy_by_layer = (
                input_map.groupby(col_layer_im)[col_energy].first().to_dict()
            )

    energies_out: list[float] = []
    ic_beam_mean: dict[str, list[float]] = {ic: [] for ic in ic_keys}
    ic_disp_mean: dict[str, list[float]] = {ic: [] for ic in ic_keys}
    ic_beam_samples: dict[str, list[np.ndarray]] = {ic: [] for ic in ic_keys}

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

        beam_off = _detect_beam_off_mask(df)
        beam_on = ~beam_off if beam_off is not None else None

        for ic in ic_keys:
            cur_cols = [c for c in _IC_CURRENT_COLS[ic] if c in df.columns]
            if not cur_cols:
                ic_beam_mean[ic].append(np.nan)
                ic_disp_mean[ic].append(np.nan)
                ic_beam_samples[ic].append(np.array([]))
                continue
            sig = df[cur_cols].sum(axis=1).to_numpy(dtype=np.float64, na_value=0.0)
            sig[~np.isfinite(sig)] = 0.0

            if beam_on is None or beam_on.sum() < MIN_BEAM_SAMPLES:
                ic_beam_mean[ic].append(np.nan)
                ic_disp_mean[ic].append(np.nan)
                ic_beam_samples[ic].append(np.array([]))
                continue

            on_samples = sig[beam_on]

            # Ratio mean: full hardware beam-on window.  ICs are phase-
            # shifted ~1-2 samples; cross-IC filtering is invalid but the
            # inter-spot zeros cancel identically in the ratio.
            ic_beam_mean[ic].append(float(np.mean(on_samples)))

            # Display: self-reflected p95 floor isolates the beam-current
            # cluster for heatmap and curve visualisation only.
            p95 = float(np.nanpercentile(on_samples, 95))
            floor = max(5.0, p95 * DISPLAY_P95_FRAC)
            beam_cluster = on_samples[on_samples >= floor]
            if len(beam_cluster) >= 3:
                ic_disp_mean[ic].append(float(np.mean(beam_cluster)))
                ic_beam_samples[ic].append(beam_cluster)
            else:
                ic_disp_mean[ic].append(np.nan)
                ic_beam_samples[ic].append(np.array([]))

    if not energies_out:
        return None

    energy_arr = np.array(energies_out, dtype=float)

    result: dict = {"energy": pd.Series(energies_out, dtype=float)}
    for ic in ic_keys:
        raw = np.array(ic_beam_mean[ic], dtype=float)
        filt = _lowpass(raw, energy_arr)
        result[f"{ic}_raw"] = raw
        result[f"{ic}_filt"] = filt

        disp = np.array(ic_disp_mean[ic], dtype=float)
        disp_filt = _lowpass(disp, energy_arr)
        result[f"{ic}_disp"] = disp
        result[f"{ic}_disp_filt"] = disp_filt
        result[f"{ic}_beam_samples"] = ic_beam_samples[ic]

    # Ratios use the plateau-beam means so they are self-consistent with
    # the curves drawn on top of the heatmaps.  The plateau ratio is the
    # steady-state gain ratio and is not distorted by per-IC response-shape
    # differences during spot ramp-up / ramp-down.
    v1_raw = result["ic1_disp"]
    v2_raw = result["ic2_disp"]
    v1_filt = result["ic1_disp_filt"]
    v2_filt = result["ic2_disp_filt"]

    result["ic21_raw"] = _sym_pct(v2_raw, v1_raw)
    result["ic21_filt"] = _sym_pct(v2_filt, v1_filt)

    if "ic3" in ic_keys:
        v3_raw = result["ic3_disp"]
        v3_filt = result["ic3_disp_filt"]
        result["ic31_raw"] = _sym_pct(v3_raw, v1_raw)
        result["ic31_filt"] = _sym_pct(v3_filt, v1_filt)
        result["ic32_raw"] = _sym_pct(v3_raw, v2_raw)
        result["ic32_filt"] = _sym_pct(v3_filt, v2_filt)

    return result


# ── Plotting helpers ─────────────────────────────────────────────────────


_RATIO_PAIRS = [
    ("ic21", "IC2 / IC1", "ic1", "ic2"),
    ("ic31", "IC3 / IC1", "ic1", "ic3"),
    ("ic32", "IC3 / IC2", "ic2", "ic3"),
]

_HEATMAP_ICS = ["ic1", "ic2", "ic3"]
_IC_MARKERS = {"ic1": "o", "ic2": "s", "ic3": "D"}


def _plot_heatmap(ax, session_data, ic_name: str, loaded_ids, sess_colors):
    """Beam-on current density heatmap with Butterworth-filtered mean overlay."""
    samples_key = f"{ic_name}_beam_samples"
    filt_key = f"{ic_name}_disp_filt"

    combined: dict[float, list[float]] = {}
    for sid in loaded_ids:
        data = session_data.get(sid)
        if data is None:
            continue
        energies = np.asarray(data["energy"], dtype=float)
        samples_list = data.get(samples_key, [])
        for i, e in enumerate(energies):
            if not np.isfinite(e):
                continue
            if i < len(samples_list) and len(samples_list[i]) > 0:
                combined.setdefault(e, []).extend(samples_list[i])

    if not combined:
        ax.set_ylabel(f"{ic_name.upper()} (nA)")
        ax.grid(visible=True, alpha=0.3)
        return

    unique_e = np.sort(np.array(list(combined.keys())))
    if len(unique_e) < 2:
        ax.set_ylabel(f"{ic_name.upper()} (nA)")
        return

    all_vals = np.concatenate(list(combined.values()))
    y_lo = max(0.0, np.nanpercentile(all_vals, 1) * 0.8)
    y_hi = np.nanpercentile(all_vals, 99) * 1.15
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0
    y_edges = np.linspace(y_lo, y_hi, HEATMAP_BINS + 1)
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    density = np.zeros((len(unique_e), HEATMAP_BINS))
    for col_i, e in enumerate(unique_e):
        vals = np.array(combined[e])
        h, _ = np.histogram(vals, bins=y_edges)
        total = h.sum()
        if total > 0:
            density[col_i] = h / total

    im = ax.pcolormesh(
        unique_e,
        y_centers,
        density.T,
        cmap="turbo",
        shading="nearest",
        zorder=1,
    )
    ax.figure.colorbar(im, ax=ax, pad=0.02, aspect=30, label="P(I)")

    for si, sid in enumerate(loaded_ids):
        data = session_data.get(sid)
        if data is None:
            continue
        e = np.asarray(data["energy"], dtype=float)
        filt = np.asarray(data.get(filt_key, []), dtype=float)
        ok = np.isfinite(e) & np.isfinite(filt)
        if ok.sum() < 2:
            continue
        sort = np.argsort(e[ok])
        ax.plot(
            e[ok][sort],
            filt[ok][sort],
            color=sess_colors[si],
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            zorder=4,
            label=sid,
        )

    ax.set_ylabel(f"{ic_name.upper()} Beam Current (nA)")
    ax.set_facecolor("black")
    ax.legend(fontsize=7, loc="upper left")


def _plot_ic_pair(ax, session_data, ic_a, ic_b, loaded_ids, sess_colors):
    """Overlay the two filtered IC curves that form a ratio pair."""
    n_sessions = len(loaded_ids)
    for si, sid in enumerate(loaded_ids):
        data = session_data.get(sid)
        if data is None:
            continue
        e = np.asarray(data["energy"], dtype=float)
        for ic_name in (ic_a, ic_b):
            disp = np.asarray(data.get(f"{ic_name}_disp_filt", []), dtype=float)
            ok = np.isfinite(e) & np.isfinite(disp)
            if not ok.any():
                continue
            marker = _IC_MARKERS.get(ic_name, "o")
            label = f"{ic_name.upper()} ({sid})" if n_sessions > 1 else ic_name.upper()
            ax.scatter(
                e[ok],
                disp[ok],
                c=sess_colors[si],
                s=14,
                alpha=0.7,
                marker=marker,
                edgecolors="none",
                zorder=3,
                label=label,
            )
    ax.set_ylabel("Beam Current (nA)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(visible=True, alpha=0.3)


def _plot_ratio(ax, session_data, ratio_key, loaded_ids, colors):
    """Scatter + robust trend line of ratio vs energy."""
    n_sessions = len(loaded_ids)
    slope_labels: list[tuple[str, tuple]] = []

    for si, sid in enumerate(loaded_ids):
        data = session_data[sid]
        if ratio_key not in data:
            continue
        prefix = f"{sid}: " if n_sessions > 1 else ""
        res = add_scatter_trend(
            ax,
            data["energy"],
            data[ratio_key],
            color=colors[si],
            unit="%/MeV",
            prefix=prefix,
            alpha=0.85,
            size=18,
            robust=True,
            outlier_sigma=OUTLIER_SIGMA,
            outlier_iterations=OUTLIER_ITERATIONS,
            highlight_rejected=True,
            show_delta=True,
        )
        if res is not None:
            slope_labels.append(res)

    if slope_labels:
        annotate_slopes(ax, slope_labels)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.grid(visible=True, alpha=0.3)


# ── Main entry point ─────────────────────────────────────────────────────

_COL_DEFS = [
    ("heatmap", "Beam-On Distribution", 3),
    ("ic_pair", "IC Curves (filtered)", 3),
    ("raw", "\u0394% (raw)", 3),
    ("filt", "\u0394% (filtered)", 3),
]


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
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    has_ic3 = any("ic31_raw" in d for d in session_data.values())
    n_ratio_rows = 3 if has_ic3 else 1
    n_heatmap_rows = 3 if has_ic3 else 2
    n_rows = max(n_ratio_rows, n_heatmap_rows)
    n_cols = len(_COL_DEFS)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 3.5 * n_rows + 1.0),
        squeeze=False,
        width_ratios=[w for _, _, w in _COL_DEFS],
    )
    fig.suptitle("Current Ratios vs Energy  (plateau mean)", **SUPTITLE_KW)

    session_data_g3 = {k: v for k, v in session_data.items() if "ic31_raw" in v}
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

        for ci, (col_key, col_label, _) in enumerate(_COL_DEFS):
            ax = axes[row, ci]

            if col_key == "heatmap":
                _plot_heatmap(ax, sdata, _HEATMAP_ICS[row], sids, scols)
            elif col_key == "ic_pair":
                if has_ratio and sdata:
                    _plot_ic_pair(ax, sdata, ic_a, ic_b, sids, scols)
                    ax.set_ylabel(f"{pair_label}  (nA)")
                else:
                    ax.set_visible(False)
            elif col_key in ("raw", "filt"):
                if has_ratio and sdata:
                    ratio_key = f"{pair_key}_{col_key}"
                    _plot_ratio(ax, sdata, ratio_key, sids, scols)
                    ax.set_ylabel(f"{pair_label}  {col_label}")
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

    legend_ax = next((ax for ax in pct_axes if ax.get_visible()), axes[0, 0])
    make_session_legend(legend_ax, loaded_ids, colors)

    apply_tight_layout()
    plt.show()
