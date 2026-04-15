"""IC timeslice audio player and WAV export.

The timeslice sample rate is 1 kHz which falls in the low audible range.
Each IC channel is normalized to [-1, 1] and can be played back directly
or saved as a 16-bit mono WAV file.

Click on any waveform to seek; drag to scrub.  The playback cursor is
shared across all IC plots.
"""

from __future__ import annotations

import time
import tkinter as tk
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..common import (
    C_IC1_CURRENT,
    C_IC2_CURRENT,
    C_IC3_CURRENT_A,
    C_IC3_CURRENT_B,
    C_IC3_CURRENT_C,
    C_IC3_CURRENT_D,
    resolve_concept_column,
)
from ..common.session_source import (
    load_session_timeslice_device_units,
    resolve_session_source,
)

FS_HZ = 1000  # timeslice period is 1 ms → 1 kHz sample rate


def _resolve_col(columns, concept: str) -> str | None:
    return resolve_concept_column(columns, concept)


def _load_ic_signals(session_id: str, base_dir: str) -> dict | None:
    """Load and concatenate all timeslice IC current arrays."""
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

    result: dict = {
        "ic1": np.concatenate(ic1_parts),
        "ic2": np.concatenate(ic2_parts),
        "has_ic3": has_ic3,
        "n_samples": n,
    }
    if has_ic3:
        result["ic3"] = np.concatenate(ic3_parts)
    return result


def _normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1] float32 for playback."""
    sig = np.nan_to_num(signal).astype(np.float64)
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak
    return sig.astype(np.float32)


_SPECSUB_ALPHA = 3.0   # oversubtraction factor — higher = more aggressive
_SPECSUB_BETA = 0.02   # spectral floor — minimum fraction of original magnitude kept

# Multi-resolution passes: each tuple is (nperseg, noverlap).  Alternating
# between a short window (good time resolution for narrow beam pulses) and
# a longer window (better frequency resolution for tonal residuals) lets
# successive passes mop up what the previous resolution missed.
_SPECSUB_PASSES: list[tuple[int, int]] = [
    (32, 24),   # 32 ms window, 75 % overlap — fine time resolution
    (64, 48),   # 64 ms window, 75 % overlap — fine frequency resolution
    (32, 24),   # second short-window pass for remaining transients
]


def _specsub_pass(
    signal: np.ndarray,
    ref_ac: np.ndarray,
    nperseg: int,
    noverlap: int,
) -> np.ndarray:
    """One pass of reference-based spectral subtraction."""
    from scipy.signal import stft, istft

    n = len(signal)
    _, _, Zt = stft(signal, fs=FS_HZ, nperseg=nperseg, noverlap=noverlap)
    _, _, Zr = stft(ref_ac, fs=FS_HZ, nperseg=nperseg, noverlap=noverlap)

    mag_t = np.abs(Zt)
    mag_r = np.abs(Zr)

    E_t = np.sum(mag_t ** 2, axis=0, keepdims=True)
    E_r = np.sum(mag_r ** 2, axis=0, keepdims=True)
    alpha_frame = np.sqrt(E_t / (E_r + 1e-20))

    beam_estimate = _SPECSUB_ALPHA * alpha_frame * mag_r
    clean_mag = np.maximum(mag_t - beam_estimate, _SPECSUB_BETA * mag_t)

    Z_clean = clean_mag * np.exp(1j * np.angle(Zt))
    _, out = istft(Z_clean, fs=FS_HZ, nperseg=nperseg, noverlap=noverlap)
    return out[:n]


def _beam_subtract(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remove beam-correlated content from *target* via spectral subtraction.

    Multi-resolution, multi-pass reference-based spectral subtraction
    (Boll 1979) adapted for IC data:

    Each pass STFTs the current residual and the reference, estimates the
    beam magnitude per time-frequency bin using a per-frame energy ratio,
    subtracts it (with a spectral floor to prevent artefacts), and
    reconstructs via inverse-STFT.  Successive passes at alternating
    window sizes catch what the previous resolution missed — short windows
    handle narrow beam transients while longer windows resolve tonal
    residuals more precisely.

    This is non-destructive: signal content that does not correlate with
    the reference in a given time-frequency bin is preserved.
    """
    t = np.nan_to_num(target).astype(np.float64)
    r = np.nan_to_num(reference).astype(np.float64)
    n = min(len(t), len(r))
    t, r = t[:n], r[:n]

    t_mean = np.mean(t)
    cur = t - t_mean
    r_ac = r - np.mean(r)

    for nperseg, noverlap in _SPECSUB_PASSES:
        cur = _specsub_pass(cur, r_ac, nperseg, noverlap)

    return cur + t_mean


def _write_wav(path: Path, signal_f32: np.ndarray, sample_rate: int) -> None:
    """Write a normalized float32 signal as 16-bit mono WAV."""
    samples = np.clip(signal_f32 * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


# ---------------------------------------------------------------------------
# Waveform envelope for fast rendering
# ---------------------------------------------------------------------------

_ENVELOPE_BINS = 2000


def _envelope(signal: np.ndarray, n_bins: int = _ENVELOPE_BINS):
    """Compute a min/max envelope for plotting long signals efficiently."""
    n = len(signal)
    if n <= n_bins * 2:
        t = np.arange(n) / FS_HZ
        return t, signal, signal
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    starts = edges[:-1]
    t = (starts + edges[1:]) * 0.5 / FS_HZ
    y_min = np.minimum.reduceat(signal, starts)
    y_max = np.maximum.reduceat(signal, starts)
    return t, y_min, y_max


# ---------------------------------------------------------------------------
# Tkinter audio player window
# ---------------------------------------------------------------------------

_BG = "#0d1117"
_FG = "#c9d1d9"
_ACCENT = "#00d4aa"
_ACCENT_DIM = "#007755"
_CURSOR_INACTIVE = "#3a3f47"
_BTN_BG = "#161b22"
_BTN_ACTIVE = "#1a2233"
_CURSOR_UPDATE_MS = 40  # ~25 fps cursor refresh

IC_COLORS = [
    "#1f77b4",  # IC1 — blue
    "#d62728",  # IC2 — red
    "#2ca02c",  # IC3 — green
    "#6baed6",  # IC1−beam — light blue
    "#fc9272",  # IC2−beam — light red
]


class _AudioPlayerWindow:
    """Tk toplevel with interactive waveform timeline and per-IC playback."""

    def __init__(self, title: str, channels: list[tuple[str, np.ndarray]]):
        self._channels = channels
        self._channel_map: dict[str, np.ndarray] = dict(channels)
        self._duration = max(len(s) for _, s in channels) / FS_HZ

        self._playing_key: str | None = None
        self._play_start_time: float = 0.0   # audio-timeline position at play start (s)
        self._wall_origin: float = 0.0        # monotonic clock snapshot at play start
        self._play_n_samples: int = 0         # number of samples queued for playback
        self._cursor_pos: float = 0.0         # current cursor time in seconds
        self._dragging = False
        self._was_playing: str | None = None  # label of channel playing before drag
        self._cursor_timer_id: str | None = None

        self._root = tk.Tk()
        self._root.title(title)
        self._root.configure(bg=_BG)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # -- waveform plots ----------------------------------------------------
        n = len(channels)
        self._fig, self._axes = plt.subplots(
            n, 1, figsize=(14, 2.2 * n), facecolor=_BG,
            squeeze=False, sharex=True,
        )
        self._fig.subplots_adjust(
            left=0.05, right=0.98, top=0.95, bottom=0.06, hspace=0.25,
        )

        self._cursor_lines: list = []
        self._ax_to_label: dict[object, str] = {}
        self._label_to_idx: dict[str, int] = {}
        for i, (label, sig) in enumerate(channels):
            ax = self._axes[i, 0]
            self._ax_to_label[ax] = label
            self._label_to_idx[label] = i
            t, y_min, y_max = _envelope(sig)
            color = IC_COLORS[i % len(IC_COLORS)]
            ax.fill_between(t, y_min, y_max, color=color, alpha=0.7)
            ax.set_ylabel(label, fontsize=9, color=_FG)
            ax.set_facecolor(_BG)
            ax.tick_params(colors=_FG, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#30363d")
            line = ax.axvline(0, color=_CURSOR_INACTIVE, linewidth=1.0,
                              alpha=0.5)
            self._cursor_lines.append(line)

        self._axes[-1, 0].set_xlabel("Time (s)", fontsize=9, color=_FG)

        self._selected_label: str = channels[0][0]
        self._update_cursor_colors()

        canvas = FigureCanvasTkAgg(self._fig, master=self._root)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas = canvas

        # -- matplotlib mouse events for click-to-seek / drag-to-scrub --------
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_release)

        # -- control bar -------------------------------------------------------
        ctrl = tk.Frame(self._root, bg=_BG)
        ctrl.pack(fill=tk.X, padx=8, pady=(4, 8))

        self._make_btn(ctrl, "\u25B6 Play", self._play_selected)
        self._make_btn(ctrl, "\u25A0 Stop", self._stop_all)
        self._make_btn(ctrl, "Save WAV", self._save_selected)

        # time display
        self._time_var = tk.StringVar(value="0.0 s")
        tk.Label(ctrl, textvariable=self._time_var, fg=_ACCENT, bg=_BG,
                 font=("Consolas", 10), width=12, anchor="e").pack(
            side=tk.RIGHT, padx=(8, 0))

        self._status_var = tk.StringVar(value="Click waveform to seek")
        tk.Label(ctrl, textvariable=self._status_var, fg=_ACCENT_DIM, bg=_BG,
                 font=("Consolas", 8), anchor="w").pack(
            side=tk.RIGHT, fill=tk.X, expand=True, padx=8)

        self._set_cursor(0.0)
        canvas.draw()

    def _make_btn(self, parent, text, command) -> tk.Button:
        btn = tk.Button(
            parent, text=text, command=command,
            fg=_ACCENT, bg=_BTN_BG, activebackground=_BTN_ACTIVE,
            activeforeground=_ACCENT, relief=tk.FLAT, padx=10, pady=2,
            font=("Consolas", 9),
        )
        btn.pack(side=tk.LEFT, padx=4)
        return btn

    # -- channel selection / cursor --------------------------------------------

    def _select_channel(self, label: str) -> None:
        if label == self._selected_label:
            return
        self._selected_label = label
        self._update_cursor_colors()
        self._canvas.draw_idle()
        self._status_var.set(f"Selected {label}")

    def _update_cursor_colors(self) -> None:
        sel_idx = self._label_to_idx.get(self._selected_label, 0)
        for i, line in enumerate(self._cursor_lines):
            if i == sel_idx:
                line.set_color(_ACCENT)
                line.set_linewidth(1.4)
                line.set_alpha(0.9)
            else:
                line.set_color(_CURSOR_INACTIVE)
                line.set_linewidth(0.8)
                line.set_alpha(0.5)

    def _set_cursor(self, time_s: float) -> None:
        time_s = max(0.0, min(time_s, self._duration))
        self._cursor_pos = time_s
        for line in self._cursor_lines:
            line.set_xdata([time_s])
        self._time_var.set(f"{time_s:.2f} s")
        self._canvas.draw_idle()

    # -- mouse events (seek / scrub) ------------------------------------------

    def _time_from_event(self, event) -> float | None:
        if event.inaxes is None or event.xdata is None:
            return None
        return float(event.xdata)

    def _on_press(self, event) -> None:
        if event.button != 1:
            return
        t = self._time_from_event(event)
        if t is None:
            return
        # Select channel by clicking its subplot
        clicked_label = self._ax_to_label.get(event.inaxes)
        if clicked_label and clicked_label != self._selected_label:
            self._select_channel(clicked_label)
        self._dragging = True
        self._was_playing = self._playing_key
        if self._was_playing:
            self._stop_all()
        self._set_cursor(t)

    def _on_motion(self, event) -> None:
        if not self._dragging:
            return
        t = self._time_from_event(event)
        if t is not None:
            self._set_cursor(t)

    def _on_release(self, event) -> None:
        if not self._dragging:
            return
        self._dragging = False
        t = self._time_from_event(event)
        if t is not None:
            self._set_cursor(t)
        if self._was_playing:
            self._play(self._selected_label, self._cursor_pos)
        self._was_playing = None

    # -- playback -------------------------------------------------------------

    def _play_selected(self) -> None:
        self._play(self._selected_label, self._cursor_pos)

    def _play(self, label: str, start_time: float) -> None:
        self._stop_all()

        sig = self._channel_map.get(label)
        if sig is None:
            return

        start_sample = int(start_time * FS_HZ)
        start_sample = max(0, min(start_sample, len(sig) - 1))
        remaining = sig[start_sample:]
        if len(remaining) == 0:
            return

        self._playing_key = label
        self._play_start_time = start_time
        self._play_n_samples = len(remaining)

        sd.play(remaining, samplerate=FS_HZ)
        self._wall_origin = time.monotonic()

        self._status_var.set(f"Playing {label}")
        self._cursor_timer_id = self._root.after(
            _CURSOR_UPDATE_MS, self._tick_cursor,
        )

    def _tick_cursor(self) -> None:
        """Periodic cursor update driven by wall-clock elapsed time."""
        if self._playing_key is None:
            return

        elapsed = time.monotonic() - self._wall_origin
        play_duration = self._play_n_samples / FS_HZ

        if elapsed >= play_duration:
            self._set_cursor(self._play_start_time + play_duration)
            self._on_playback_done()
            return

        self._set_cursor(self._play_start_time + elapsed)
        self._cursor_timer_id = self._root.after(
            _CURSOR_UPDATE_MS, self._tick_cursor,
        )

    def _on_playback_done(self) -> None:
        self._playing_key = None
        self._status_var.set("Click waveform to seek")
        self._cursor_timer_id = None

    def _stop_all(self) -> None:
        self._playing_key = None
        if self._cursor_timer_id is not None:
            self._root.after_cancel(self._cursor_timer_id)
            self._cursor_timer_id = None
        sd.stop()
        self._status_var.set("Stopped")

    # -- save -----------------------------------------------------------------

    def _save_selected(self) -> None:
        from tkinter import filedialog
        label = self._selected_label
        sig = self._channel_map.get(label)
        if sig is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            initialfile=f"{label}.wav",
        )
        if path:
            _write_wav(Path(path), sig, FS_HZ)
            self._status_var.set(f"Saved {Path(path).name}")

    # -- lifecycle ------------------------------------------------------------

    def _on_close(self) -> None:
        self._stop_all()
        plt.close(self._fig)
        self._root.destroy()
        self._root.quit()

    def mainloop(self) -> None:
        self._root.mainloop()


# ---------------------------------------------------------------------------
# View entry point
# ---------------------------------------------------------------------------

def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Launch the IC audio player for the selected sessions."""
    if not session_ids:
        print("No sessions selected")
        return

    for sid in session_ids:
        data = _load_ic_signals(sid, base_dir)
        if data is None:
            print(f"  {sid}: no timeslice data found, skipping")
            continue

        raw_ic1 = data.get("ic1")
        raw_ic2 = data.get("ic2")
        raw_ic3 = data.get("ic3")

        channels: list[tuple[str, np.ndarray]] = []
        if raw_ic1 is not None:
            channels.append((f"{sid} IC1", _normalize(raw_ic1)))
        if raw_ic2 is not None:
            channels.append((f"{sid} IC2", _normalize(raw_ic2)))
        if raw_ic3 is not None:
            channels.append((f"{sid} IC3", _normalize(raw_ic3)))

        # IC3 is acoustically shielded → use it as a beam reference to
        # produce corrected IC1/IC2 with beam pulses removed.
        if raw_ic3 is not None:
            if raw_ic1 is not None:
                corr1 = _beam_subtract(raw_ic1, raw_ic3)
                channels.append((f"{sid} IC1\u2212beam", _normalize(corr1)))
            if raw_ic2 is not None:
                corr2 = _beam_subtract(raw_ic2, raw_ic3)
                channels.append((f"{sid} IC2\u2212beam", _normalize(corr2)))

        if not channels:
            continue

        duration = data["n_samples"] / FS_HZ
        title = f"IC Audio — {sid}  ({duration:.1f}s @ {FS_HZ} Hz)"
        player = _AudioPlayerWindow(title, channels)
        player.mainloop()
