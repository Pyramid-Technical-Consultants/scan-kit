"""IC current FFT analysis: frequency-domain view of timeslice currents."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.signal import find_peaks

from ..common import (
    C_IC1_CURRENT,
    C_IC2_CURRENT,
    C_IC3_CURRENT_A,
    C_IC3_CURRENT_B,
    C_IC3_CURRENT_C,
    C_IC3_CURRENT_D,
    resolve_concept_column,
    DEFAULT_SESSION_COLORS,
    SUPTITLE_KW,
    GRID_KW,
)
from ..common.session_source import (
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

ON_FRAC = 0.10
OFF_FRAC = 0.02


def _resolve_col(columns, concept: str) -> str | None:
    return resolve_concept_column(columns, concept)


def _classify_timeslices(signal: np.ndarray):
    """Return boolean masks ``(beam_on, beam_off)`` based on signal amplitude.

    Uses the same threshold logic as the beam-on/off current view.
    """
    bg = np.nanpercentile(signal, 25)
    pk = np.nanpercentile(signal, 99)
    dyn = pk - bg
    if pk == 0 or abs(dyn / pk) < 0.05:
        return None, None
    on_mask = signal > (bg + ON_FRAC * dyn)
    off_mask = signal < (bg + OFF_FRAC * dyn)
    return on_mask, off_mask


def _load_ic_signals(session_id: str, base_dir: str) -> dict | None:
    """Load timeslice IC currents and split into beam-on / background."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    frames = load_session_timeslice_device_units(src)
    if not frames:
        return None

    df0 = frames[0]
    ts_ic1 = _resolve_col(df0.columns, C_IC1_CURRENT)
    ts_ic2 = _resolve_col(df0.columns, C_IC2_CURRENT)
    if not all([ts_ic1, ts_ic2]):
        return None

    ts_ic3a = _resolve_col(df0.columns, C_IC3_CURRENT_A)
    ts_ic3b = _resolve_col(df0.columns, C_IC3_CURRENT_B)
    ts_ic3c = _resolve_col(df0.columns, C_IC3_CURRENT_C)
    ts_ic3d = _resolve_col(df0.columns, C_IC3_CURRENT_D)
    has_ic3 = all([ts_ic3a, ts_ic3b, ts_ic3c, ts_ic3d])

    ic1_parts: list[np.ndarray] = []
    ic2_parts: list[np.ndarray] = []
    ic3_parts: list[np.ndarray] = []

    for df in frames:
        ic1_parts.append(df[ts_ic1].values)
        ic2_parts.append(df[ts_ic2].values)
        if has_ic3:
            ic3_parts.append(
                df[ts_ic3a].values
                + df[ts_ic3b].values
                + df[ts_ic3c].values
                + df[ts_ic3d].values
            )

    n = sum(len(p) for p in ic1_parts)
    if n == 0:
        return None

    signals: dict[str, np.ndarray] = {
        "ic1": np.concatenate(ic1_parts),
        "ic2": np.concatenate(ic2_parts),
    }
    if has_ic3:
        signals["ic3"] = np.concatenate(ic3_parts)

    result: dict = {"has_ic3": has_ic3, "n_samples": n}
    for key, sig in signals.items():
        on_mask, off_mask = _classify_timeslices(sig)
        if on_mask is None:
            result[f"{key}_on"] = np.array([])
            result[f"{key}_off"] = np.array([])
        else:
            result[f"{key}_on"] = sig[on_mask]
            result[f"{key}_off"] = sig[off_mask]
    return result


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


def _style_fft_ax(ax: plt.Axes, ic_label: str, is_bottom: bool) -> None:
    """Apply shared axis styling to an FFT subplot."""
    ax.set_xlim(FREQ_MIN_HZ, FREQ_MAX_HZ)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.tick_params(which="minor", length=3)
    ax.set_ylabel(f"{ic_label}\nPSD (nA²/Hz)", fontsize=9)
    ax.grid(**GRID_KW)
    ax.grid(which="minor", color="#e0e0e0", linewidth=0.3, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    if is_bottom:
        ax.set_xlabel("Frequency (Hz)", fontsize=10)


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Launch the IC current FFT analysis viewer."""
    if not session_ids:
        print("No sessions selected")
        return

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        data = _load_ic_signals(sid, base_dir)
        if data is not None:
            session_data[sid] = data

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

    fig, axes = plt.subplots(
        n_ics, 2,
        figsize=(22, 3.5 * n_ics + 1),
        squeeze=False,
    )
    fig.suptitle("IC Current FFT Analysis", **SUPTITLE_KW)
    axes[0, 0].set_title("Background", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Beam", fontsize=11, fontweight="bold")

    # X-axis: all 6 subplots share one common x-axis so zoom syncs everywhere.
    # Y-axis: each column shares its own y-axis so the 3 background rows are
    # directly comparable, and the 3 beam rows are directly comparable, but
    # background and beam have independent scales.
    ax_ref = axes[0, 0]
    for row in range(n_ics):
        for col in range(2):
            if (row, col) == (0, 0):
                continue
            axes[row, col].sharex(ax_ref)
    for row in range(1, n_ics):
        axes[row, 0].sharey(axes[0, 0])
        axes[row, 1].sharey(axes[0, 1])

    for ic_idx, (ic_key, ic_label) in enumerate(zip(ic_keys, ic_labels)):
        ax_off = axes[ic_idx, 0]
        ax_on = axes[ic_idx, 1]
        is_bottom = ic_idx == n_ics - 1

        for si, (sid, data) in enumerate(session_data.items()):
            label = sid if multi else ic_label
            color = sess_colors[si] if multi else IC_COLORS[ic_idx]

            for ax, suffix in ((ax_off, "_off"), (ax_on, "_on")):
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
                    linewidth=0.6, alpha=0.85, label=label, color=color,
                )
                _annotate_peaks(ax, freqs_hz[band], psd[band], color)

        _style_fft_ax(ax_off, ic_label, is_bottom)
        _style_fft_ax(ax_on, ic_label, is_bottom)

    for row in range(n_ics - 1):
        for col in range(2):
            plt.setp(axes[row, col].get_xticklabels(), visible=False)

    fig.subplots_adjust(
        top=0.91, bottom=0.06, left=0.05, right=0.98, hspace=0.15, wspace=0.08,
    )
    plt.show()
