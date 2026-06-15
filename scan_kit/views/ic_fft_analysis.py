"""IC current FFT analysis: frequency-domain view of timeslice currents."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

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
    DEFAULT_SESSION_COLORS,
    finish_view,
    VIEW_HEADER_SUBPLOT_TOP,
    GRID_KW,
)
from ..common.processing import _detect_beam_off_mask
from ..common.session_source import (
    load_session_csv,
    load_session_timeslice_device_units,
    resolve_session_source,
)

FS_HZ = 1000.0  # timeslice period is 1 ms → 1 kHz sample rate
SEGMENT_LENGTH = 4096
OVERLAP_FRACTION = 0.5
FREQ_MIN_HZ = 1.0
FREQ_MAX_HZ = 500.0
PEAK_PROMINENCE_FACTOR = 20.0
MAX_PEAKS_PER_IC = 8

_BG_NOISE_FLOOR_NA = 10.0
_BG_GUARD_SAMPLES = 3  # 3 ms at 1 kHz


def _resolve_col(columns, concept: str) -> str | None:
    return resolve_concept_column(columns, concept)


def _beam_on_mask(df, *, strict: bool = False) -> np.ndarray:
    """Boolean mask: True where the hardware says beam is on."""
    off = _detect_beam_off_mask(df, strict=strict)
    if off is not None:
        return ~off
    return np.ones(len(df), dtype=bool)


def _current_quiet_mask(sig: np.ndarray) -> np.ndarray:
    """True where *sig* is below the noise floor and not near a hot sample."""
    hot = np.abs(sig) > _BG_NOISE_FLOOR_NA
    if _BG_GUARD_SAMPLES > 0:
        from scipy.ndimage import binary_dilation
        hot = binary_dilation(hot, iterations=_BG_GUARD_SAMPLES)
    return ~hot


def _extract_ic_signals(df, ts_ic1, ts_ic2, ts_ic3_cols, has_ic3):
    """Return dict of IC arrays from a single frame."""
    sigs: dict[str, np.ndarray] = {
        "ic1": df[ts_ic1].values.astype(float),
        "ic2": df[ts_ic2].values.astype(float),
    }
    if has_ic3:
        sigs["ic3"] = sum(
            df[c].values.astype(float) for c in ts_ic3_cols
        )
    return sigs


def _resolve_ic_columns(df0):
    """Resolve IC column names. Returns (ic1, ic2, ic3_cols, has_ic3)."""
    ts_ic1 = _resolve_col(df0.columns, C_IC1_CURRENT)
    ts_ic2 = _resolve_col(df0.columns, C_IC2_CURRENT)
    ic3_cols = [
        _resolve_col(df0.columns, c)
        for c in (C_IC3_CURRENT_A, C_IC3_CURRENT_B, C_IC3_CURRENT_C, C_IC3_CURRENT_D)
    ]
    has_ic3 = all(ic3_cols)
    return ts_ic1, ts_ic2, ic3_cols, has_ic3


def _load_ic_signals(session_id: str, base_dir: str, *, bg_subtract: bool = False) -> dict | None:
    """Load timeslice IC currents and split into beam-on / background."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    frames = load_session_timeslice_device_units(src)
    if not frames:
        return None
    if bg_subtract:
        from ..common import subtract_background_frames
        subtract_background_frames(frames)

    ts_ic1, ts_ic2, ic3_cols, has_ic3 = _resolve_ic_columns(frames[0])
    if not all([ts_ic1, ts_ic2]):
        return None

    on_parts: dict[str, list] = {"ic1": [], "ic2": []}
    off_parts: dict[str, list] = {"ic1": [], "ic2": []}
    if has_ic3:
        on_parts["ic3"] = []
        off_parts["ic3"] = []

    for df in frames:
        hw_on = _beam_on_mask(df)
        sigs = _extract_ic_signals(df, ts_ic1, ts_ic2, ic3_cols, has_ic3)
        for key, sig in sigs.items():
            quiet = _current_quiet_mask(sig)
            on_parts[key].append(sig[hw_on])
            off_parts[key].append(sig[~hw_on & quiet])

    n = sum(len(p) for p in on_parts["ic1"]) + sum(len(p) for p in off_parts["ic1"])
    if n == 0:
        return None

    result: dict = {"has_ic3": has_ic3, "n_samples": n}
    for key in on_parts:
        result[f"{key}_on"] = np.concatenate(on_parts[key]) if on_parts[key] else np.array([])
        result[f"{key}_off"] = np.concatenate(off_parts[key]) if off_parts[key] else np.array([])
    return result


def _load_per_energy_signals(
    session_id: str, base_dir: str, *, bg_subtract: bool = False,
) -> dict | None:
    """Load timeslice IC currents split by energy layer.

    Returns a dict with ``energies`` (sorted list), ``has_ic3``, and for
    each IC key a dict mapping energy -> ``(beam_on_signal, beam_off_signal)``.
    """
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    input_map = load_session_csv(src, "input_map.csv")
    if input_map is None or C_ENERGY not in input_map.columns:
        return None

    energy_by_layer_id: dict | None = None
    energy_by_idx: dict[int, float] | None = None

    if C_LAYER_ID in input_map.columns:
        energy_by_layer_id = (
            input_map.groupby(C_LAYER_ID)[C_ENERGY].first().to_dict()
        )
    if energy_by_layer_id is None or len(energy_by_layer_id) <= 1:
        ordered = list(dict.fromkeys(input_map[C_ENERGY].values))
        energy_by_idx = {i: e for i, e in enumerate(ordered)}

    frames = load_session_timeslice_device_units(src)
    if not frames:
        return None
    if bg_subtract:
        from ..common import subtract_background_frames
        subtract_background_frames(frames)

    ts_ic1, ts_ic2, ic3_cols, has_ic3 = _resolve_ic_columns(frames[0])
    if not all([ts_ic1, ts_ic2]):
        return None

    ic_keys = ["ic1", "ic2"] + (["ic3"] if has_ic3 else [])
    per_energy: dict[str, dict[float, tuple]] = {k: {} for k in ic_keys}

    for df in frames:
        energy = None
        if energy_by_idx is not None and "_layer_idx" in df.columns:
            idx = int(df["_layer_idx"].iloc[0])
            energy = energy_by_idx.get(idx)
        if energy is None and energy_by_layer_id is not None and C_LAYER_ID in df.columns:
            lid = df[C_LAYER_ID].iloc[0]
            energy = energy_by_layer_id.get(lid)
        if energy is None:
            continue

        hw_on = _beam_on_mask(df)
        sigs = _extract_ic_signals(df, ts_ic1, ts_ic2, ic3_cols, has_ic3)

        for key, sig in sigs.items():
            quiet = _current_quiet_mask(sig)
            on_sig = sig[hw_on]
            off_sig = sig[~hw_on & quiet]
            if energy in per_energy[key]:
                on_sig = np.concatenate([per_energy[key][energy][0], on_sig])
                off_sig = np.concatenate([per_energy[key][energy][1], off_sig])
            per_energy[key][energy] = (on_sig, off_sig)

    all_energies: set[float] = set()
    for d in per_energy.values():
        all_energies.update(d.keys())
    if not all_energies:
        return None

    return {
        "energies": sorted(all_energies),
        "has_ic3": has_ic3,
        "per_energy": per_energy,
    }


def _welch_psd(signal: np.ndarray, fs: float, seg_len: int, overlap: float):
    """Estimate power spectral density via Welch's method (numpy-only).

    Returns ``(freqs_hz, psd)`` with physical frequency in Hz.
    """
    sig = np.nan_to_num(signal - np.nanmean(signal))
    n = len(sig)
    if n < seg_len:
        seg_len = max(16, n)

    step = max(1, int(seg_len * (1 - overlap)))
    window = np.hanning(seg_len)
    win_power = np.sum(window ** 2)

    psd_accum: np.ndarray | None = None
    count = 0
    for s in range(0, n - seg_len + 1, step):
        segment = sig[s: s + seg_len] * window
        power = np.abs(np.fft.rfft(segment)) ** 2
        if psd_accum is None:
            psd_accum = power
        else:
            psd_accum += power
        count += 1

    if psd_accum is None or count == 0:
        return None, None

    psd = psd_accum / (count * win_power)
    psd[1:-1] *= 2  # single-sided doubling (skip DC and Nyquist)
    freqs_hz = np.fft.rfftfreq(seg_len, d=1.0 / fs)
    return freqs_hz, psd


# ---------------------------------------------------------------------------
# View entry point
# ---------------------------------------------------------------------------

IC_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]


_LABEL_MIN_X_GAP_PX = 50  # minimum horizontal pixel distance between labels
_LABEL_OFFSET_LOW = 8     # default vertical offset (points)
_LABEL_OFFSET_HIGH = 22   # elevated offset when dodging a neighbour


def _annotate_peaks(ax, freqs: np.ndarray, psd: np.ndarray, color: str) -> None:
    """Find prominent peaks and label them, staggering to avoid overlap."""
    if len(psd) < 3:
        return
    median_psd = np.median(psd)
    prominence = median_psd * PEAK_PROMINENCE_FACTOR
    indices, props = find_peaks(psd, prominence=prominence)
    if len(indices) == 0:
        return

    order = np.argsort(props["prominences"])[::-1][:MAX_PEAKS_PER_IC]
    indices = indices[order]
    indices = np.sort(indices)  # left-to-right for overlap logic

    placed: list[tuple[float, float]] = []  # (x_display, y_offset) of placed labels

    transform = ax.transData

    for idx in indices:
        freq = freqs[idx]
        amp = psd[idx]
        x_disp = transform.transform((freq, amp))[0]

        y_off = _LABEL_OFFSET_LOW
        for prev_x, prev_off in placed:
            if abs(x_disp - prev_x) < _LABEL_MIN_X_GAP_PX:
                y_off = _LABEL_OFFSET_HIGH if prev_off == _LABEL_OFFSET_LOW else _LABEL_OFFSET_LOW
                break

        ax.annotate(
            f"{freq:.1f} Hz",
            xy=(freq, amp),
            xytext=(0, y_off), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold",
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
        )
        placed.append((x_disp, y_off))


def _style_fft_ax(ax: plt.Axes, is_bottom: bool) -> None:
    """Apply shared axis styling to an FFT subplot."""
    ax.set_xlim(FREQ_MIN_HZ, FREQ_MAX_HZ)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.tick_params(which="minor", length=3)
    ax.set_ylabel("PSD (nA²/Hz)", fontsize=9)
    ax.grid(**GRID_KW)
    ax.grid(which="minor", color="#e0e0e0", linewidth=0.3, alpha=0.5)
    if is_bottom:
        ax.set_xlabel("Frequency (Hz)", fontsize=10)


_HEATMAP_SEG_LEN = 1024
_HEATMAP_CMAP = "turbo"


def _build_psd_heatmap(
    per_energy_ic: dict[float, tuple],
    energies: list[float],
    sig_idx: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build a 2-D PSD array (energy × freq) for one IC.

    *sig_idx*: 0 for beam-on, 1 for beam-off.
    Returns ``(freqs_hz_band, psd_2d)`` or ``(None, None)``.
    """
    lengths = []
    for energy in energies:
        pair = per_energy_ic.get(energy)
        if pair is not None and len(pair[sig_idx]) >= 32:
            lengths.append(len(pair[sig_idx]))
    if not lengths:
        return None, None

    seg_len = min(_HEATMAP_SEG_LEN, min(lengths))
    seg_len = max(16, seg_len)

    rows = []
    common_freqs = None
    for energy in energies:
        pair = per_energy_ic.get(energy)
        if pair is None:
            rows.append(None)
            continue
        sig = pair[sig_idx]
        if len(sig) < 32:
            rows.append(None)
            continue
        freqs, psd = _welch_psd(sig, FS_HZ, seg_len, OVERLAP_FRACTION)
        if freqs is None:
            rows.append(None)
            continue
        band = (freqs >= FREQ_MIN_HZ) & (freqs <= FREQ_MAX_HZ)
        if common_freqs is None:
            common_freqs = freqs[band]
        rows.append(psd[band])

    if common_freqs is None:
        return None, None

    n_freq = len(common_freqs)
    psd_2d = np.full((len(energies), n_freq), np.nan)
    for i, row in enumerate(rows):
        if row is not None and len(row) == n_freq:
            psd_2d[i] = row

    return common_freqs, psd_2d


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Launch the IC current FFT analysis viewer."""
    if not session_ids:
        print("No sessions selected")
        return

    bg = settings.bg_subtract if settings else False

    session_data: dict[str, dict] = {}
    session_energy_data: dict[str, dict] = {}
    for sid in session_ids:
        data = _load_ic_signals(sid, base_dir, bg_subtract=bg)
        if data is not None:
            session_data[sid] = data
        edata = _load_per_energy_signals(sid, base_dir, bg_subtract=bg)
        if edata is not None:
            session_energy_data[sid] = edata

    if not session_data:
        print("No valid timeslice data found for any session")
        return

    loaded_ids = list(session_data.keys())
    sess_colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    multi = len(loaded_ids) > 1
    show_ic3 = any(d.get("has_ic3", False) for d in session_data.values())

    ic_keys: list[str] = ["ic1", "ic2"]
    ic_labels: list[str] = ["IC1", "IC2"]
    if show_ic3:
        ic_keys.append("ic3")
        ic_labels.append("IC3 (A+B+C+D)")

    n_ics = len(ic_keys)
    session_energy_data = {
        sid: edata for sid, edata in session_energy_data.items()
        if len(edata["energies"]) > 1
    }
    has_heatmaps = len(session_energy_data) > 0
    n_sess_hm = len(session_energy_data)

    # Layout — columns: one per IC + a narrow colorbar column
    # Rows: 1 line-plot row, then per session: beam-on heatmap, beam-off heatmap
    import matplotlib.colors as mcolors

    hm_rows_per_sess = 2  # beam-on, beam-off
    n_hm_rows = n_sess_hm * hm_rows_per_sess if has_heatmaps else 0
    n_rows = 1 + n_hm_rows
    height_ratios = [1.0] + [1.0] * n_hm_rows

    n_cols = n_ics + (1 if has_heatmaps else 0)
    width_ratios = [1] * n_ics + ([0.03] if has_heatmaps else [])

    fig = plt.figure(figsize=(7 * n_ics + 1, 3.0 * n_rows + 1))
    gs = fig.add_gridspec(
        n_rows, n_cols,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        left=0.05, right=0.97, top=VIEW_HEADER_SUBPLOT_TOP, bottom=0.04,
        hspace=0.25, wspace=0.18,
    )

    # --- Row 0: line plots (one per IC column) ---
    line_axes: list[plt.Axes] = []
    for ic_idx in range(n_ics):
        ax = fig.add_subplot(gs[0, ic_idx])
        line_axes.append(ax)
    for ic_idx in range(1, n_ics):
        line_axes[ic_idx].sharex(line_axes[0])
        line_axes[ic_idx].sharey(line_axes[0])

    for ic_idx, (ic_key, ic_label) in enumerate(zip(ic_keys, ic_labels)):
        ax = line_axes[ic_idx]
        for si, (sid, data) in enumerate(session_data.items()):
            color = sess_colors[si] if multi else IC_COLORS[ic_idx]
            for suffix, ls in (("_on", "-"), ("_off", "--")):
                signal = data.get(f"{ic_key}{suffix}")
                if signal is None or len(signal) == 0:
                    continue
                freqs_hz, psd = _welch_psd(
                    signal, FS_HZ, SEGMENT_LENGTH, OVERLAP_FRACTION,
                )
                if freqs_hz is None:
                    continue
                band = (freqs_hz >= FREQ_MIN_HZ) & (freqs_hz <= FREQ_MAX_HZ)
                ax.semilogy(
                    freqs_hz[band], psd[band],
                    linewidth=0.6, alpha=0.85,
                    color=color, linestyle=ls,
                )
                if ls == "-":
                    _annotate_peaks(ax, freqs_hz[band], psd[band], color)

        _style_fft_ax(ax, is_bottom=not has_heatmaps)
        ax.set_title(ic_label, fontsize=11, fontweight="bold")
        if has_heatmaps:
            plt.setp(ax.get_xticklabels(), visible=False)

    linestyle_handles = [
        Line2D([0], [0], color="0.35", linestyle="-", linewidth=0.9, label="Beam on"),
        Line2D([0], [0], color="0.35", linestyle="--", linewidth=0.9, label="Beam off"),
    ]
    if has_heatmaps:
        leg_ax = fig.add_subplot(gs[0, n_ics])
        leg_ax.axis("off")
        leg_ax.legend(
            linestyle_handles, loc="center", fontsize=8,
            frameon=True, framealpha=0.9,
        )
    else:
        line_axes[-1].legend(
            linestyle_handles, loc="upper right", fontsize=8,
        )

    # --- Heatmap rows ---
    if has_heatmaps:
        # Compute separate color norms for beam-on (sig_idx=0) and beam-off (sig_idx=1)
        state_norms: dict[int, mcolors.LogNorm] = {}
        for sig_idx in (0, 1):
            vmin, vmax = np.inf, -np.inf
            for edata in session_energy_data.values():
                for ic_key in ic_keys:
                    pe = edata["per_energy"].get(ic_key)
                    if pe is None:
                        continue
                    freqs, psd_2d = _build_psd_heatmap(
                        pe, edata["energies"], sig_idx,
                    )
                    if psd_2d is None:
                        continue
                    fp = psd_2d[np.isfinite(psd_2d)]
                    fp = fp[fp > 0]
                    if len(fp) > 0:
                        vmin = min(vmin, np.percentile(fp, 1))
                        vmax = max(vmax, np.percentile(fp, 99))
            if np.isfinite(vmin):
                state_norms[sig_idx] = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                state_norms[sig_idx] = mcolors.LogNorm(vmin=1e-4, vmax=1.0)

        first_im_by_state: dict[int, object] = {}
        first_row_by_state: dict[int, int] = {}

        for si, (sid, edata) in enumerate(session_energy_data.items()):
            energies = edata["energies"]
            per_energy = edata["per_energy"]

            for state_off, (sig_idx, state_lbl) in enumerate(
                ((0, "Beam"), (1, "BG")),
            ):
                row = 1 + si * hm_rows_per_sess + state_off
                is_bottom = row == n_rows - 1
                norm = state_norms[sig_idx]

                for ic_idx, (ic_key, ic_label) in enumerate(zip(ic_keys, ic_labels)):
                    ax = fig.add_subplot(gs[row, ic_idx])

                    pe = per_energy.get(ic_key)
                    if pe is None:
                        ax.text(
                            0.5, 0.5, "No data", transform=ax.transAxes,
                            ha="center", va="center", fontsize=12, color="gray",
                        )
                    else:
                        freqs, psd_2d = _build_psd_heatmap(pe, energies, sig_idx)
                        if freqs is not None and psd_2d is not None:
                            psd_2d = np.clip(psd_2d, norm.vmin, None)
                            im = ax.pcolormesh(
                                freqs, energies, psd_2d,
                                cmap=_HEATMAP_CMAP, norm=norm,
                                shading="nearest",
                            )
                            if sig_idx not in first_im_by_state:
                                first_im_by_state[sig_idx] = im
                                first_row_by_state[sig_idx] = row
                            ax.grid(color="white", linewidth=0.3, alpha=0.3)

                    ax.set_xlim(FREQ_MIN_HZ, FREQ_MAX_HZ)
                    if ic_idx == 0:
                        sess_tag = f"{sid} " if multi else ""
                        ax.set_ylabel(
                            f"{sess_tag}{state_lbl}\nEnergy (MeV)", fontsize=9,
                        )
                    if is_bottom:
                        ax.set_xlabel("Frequency (Hz)", fontsize=10)
                    else:
                        plt.setp(ax.get_xticklabels(), visible=False)

        # One colorbar per beam state, placed next to its first heatmap row
        for sig_idx, state_lbl in ((0, "Beam"), (1, "Background")):
            im = first_im_by_state.get(sig_idx)
            row = first_row_by_state.get(sig_idx)
            if im is None or row is None:
                continue
            cbar_ax = fig.add_subplot(gs[row, n_ics])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label(f"{state_lbl} PSD (nA²/Hz)", fontsize=8)

    finish_view(
        fig,
        "IC Current FFT Analysis",
        loaded_ids,
        sess_colors,
        base_dir=base_dir,
    )
