"""Timeslice signal preparation for the IC audio player."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..common.data_filter import (
    FILTER_ALL,
    FILTER_BEAM_OFF,
    filter_mask_from_columns,
)
from ..data.timeline_channels import TIMELINE_CHANNEL_BY_KEY
from .fft_catalog import CHANNEL_BY_ID, FftConfig, METRIC_IC_CURRENT
from .fft_data import FS_HZ
from .timeslice_replay_common import compress_minmax

_ENVELOPE_BINS = 2000
LIVE_FFT_WINDOW_MS = (50, 100, 250, 500, 1000)
DEFAULT_LIVE_FFT_WINDOW_MS = 250
SPECTRUM_DB_FLOOR = -80.0
SPECTRUM_FMAX_HZ = FS_HZ / 2.0

_SPECSUB_ALPHA = 3.0
_SPECSUB_BETA = 0.02
_SPECSUB_PASSES: list[tuple[int, int]] = [
    (32, 24),
    (64, 48),
    (32, 24),
]

_BEAM_SUBTRACT_CHANNELS = frozenset({"ic1", "ic2"})


@dataclass
class AudioPlayerConfig(FftConfig):
    beam_subtract: bool = False


@dataclass(frozen=True)
class PlaybackChannel:
    label: str
    channel_id: str
    signal: np.ndarray
    color: str


@dataclass(frozen=True)
class WaveformRenderChannel(PlaybackChannel):
    """Playback channel with precomputed vispy envelope and line geometry."""

    envelope_poly: np.ndarray
    line_pos: np.ndarray
    y_lo: float
    y_hi: float


CacheKey = tuple[str, str, str, str, str, bool, tuple[str, ...]]


def normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1] float32 for playback."""
    sig = np.nan_to_num(signal).astype(np.float64)
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak
    return sig.astype(np.float32)


def _effective_stft_params(n: int, nperseg: int, noverlap: int) -> tuple[int, int]:
    nperseg = min(nperseg, max(n, 1))
    noverlap = min(noverlap, max(nperseg - 1, 0))
    return nperseg, noverlap


def _hann_periodic(nperseg: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(nperseg, dtype=np.float64) / nperseg)


def _overlap_add(frames: np.ndarray, hop: int) -> np.ndarray:
    nframes, nperseg = frames.shape
    out_len = nperseg + (nframes - 1) * hop
    if nperseg % hop == 0:
        nstep = nperseg // hop
        pad = (-nframes) % nstep
        if pad:
            frames = np.vstack((frames, np.zeros((pad, nperseg), dtype=frames.dtype)))
        nblocks = frames.shape[0] // nstep
        folded = frames.reshape(nblocks, nstep, nperseg)
        y = np.zeros(nblocks * nperseg + nperseg, dtype=np.float64)
        for k in range(nstep):
            y[k * hop : k * hop + nblocks * nperseg] += folded[:, k, :].reshape(-1)
        return y[:out_len]
    y = np.zeros(out_len, dtype=np.float64)
    idx = np.arange(nperseg) + np.arange(nframes)[:, None] * hop
    np.add.at(y, idx, frames)
    return y


def _stft_spectrum(x: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
    """One-sided STFT matching ``scipy.signal.stft(..., scaling='spectrum')``."""
    hop = nperseg - noverlap
    win = _hann_periodic(nperseg)
    pad = nperseg // 2
    x_ext = np.pad(x, pad, mode="constant")
    nadd = (-(len(x_ext) - nperseg) % hop) % hop
    if nadd:
        x_ext = np.pad(x_ext, (0, nadd), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(x_ext, nperseg)[::hop]
    return np.fft.rfft(frames * win, axis=-1).T / win.sum()


def _istft_spectrum(
    z: np.ndarray, nperseg: int, noverlap: int, nout: int,
) -> np.ndarray:
    """Invert :func:`_stft_spectrum` (matches ``scipy.signal.istft``)."""
    hop = nperseg - noverlap
    win = _hann_periodic(nperseg)
    frames = np.fft.irfft(z.T * win.sum(), n=nperseg, axis=-1) * win
    y = _overlap_add(frames, hop)
    wnorm = _overlap_add(np.tile(win * win, (frames.shape[0], 1)), hop)
    nz = wnorm > 1e-10
    y[nz] /= wnorm[nz]
    pad = nperseg // 2
    return y[pad : pad + nout]


def _specsub_pass(
    signal: np.ndarray,
    zr: np.ndarray,
    nperseg: int,
    noverlap: int,
) -> np.ndarray:
    n = len(signal)
    nperseg, noverlap = _effective_stft_params(n, nperseg, noverlap)
    zt = _stft_spectrum(signal, nperseg, noverlap)

    mag_t = np.abs(zt)
    mag_r = np.abs(zr)
    e_t = np.sum(mag_t ** 2, axis=0, keepdims=True)
    e_r = np.sum(mag_r ** 2, axis=0, keepdims=True)
    alpha_frame = np.sqrt(e_t / (e_r + 1e-20))

    beam_estimate = _SPECSUB_ALPHA * alpha_frame * mag_r
    clean_mag = np.maximum(mag_t - beam_estimate, _SPECSUB_BETA * mag_t)
    z_clean = clean_mag * np.exp(1j * np.angle(zt))
    return _istft_spectrum(z_clean, nperseg, noverlap, n)


def beam_subtract(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remove beam-correlated content from *target* via spectral subtraction."""
    t = np.nan_to_num(target).astype(np.float64)
    r = np.nan_to_num(reference).astype(np.float64)
    n = min(len(t), len(r))
    t, r = t[:n], r[:n]
    if n < 2:
        return t

    t_mean = np.mean(t)
    cur = t - t_mean
    r_ac = r - np.mean(r)

    ref_spectra: dict[tuple[int, int], np.ndarray] = {}
    for nperseg, noverlap in _SPECSUB_PASSES:
        nperseg, noverlap = _effective_stft_params(n, nperseg, noverlap)
        zr = ref_spectra.get((nperseg, noverlap))
        if zr is None:
            zr = _stft_spectrum(r_ac, nperseg, noverlap)
            ref_spectra[(nperseg, noverlap)] = zr
        cur = _specsub_pass(cur, zr, nperseg, noverlap)

    return cur + t_mean


def write_wav(path: Path, signal_f32: np.ndarray, sample_rate: int = int(FS_HZ)) -> None:
    """Write a normalized float32 signal as 16-bit mono WAV."""
    samples = np.clip(signal_f32 * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def channel_color(channel_id: str, *, beam_subtracted: bool = False) -> str:
    spec = TIMELINE_CHANNEL_BY_KEY.get(channel_id)
    base = spec.replay_color if spec and spec.replay_color else "#c9d1d9"
    if not beam_subtracted:
        return base
    if channel_id == "ic1":
        return "#6baed6"
    if channel_id == "ic2":
        return "#fc9272"
    return base


def _channel_label(channel_id: str, *, beam_subtracted: bool = False) -> str:
    label = CHANNEL_BY_ID[channel_id].label if channel_id in CHANNEL_BY_ID else channel_id
    if beam_subtracted:
        return f"{label}\u2212beam"
    return label


def extract_audio_signal(
    session: dict,
    channel_id: str,
    *,
    domain_filter: str,
    beam_state_filter: str,
    filter_column_keys: Sequence[str],
    beam_off_quiet_threshold: float | None = None,
) -> np.ndarray:
    """Return a continuous filtered time-domain array suitable for playback."""
    if channel_id not in session:
        return np.array([], dtype=float)
    arrays = _mask_session_arrays(
        session,
        (channel_id,),
        domain_filter=domain_filter,
        beam_state_filter=beam_state_filter,
        filter_column_keys=filter_column_keys,
        quiet_threshold=beam_off_quiet_threshold,
    )
    return arrays[0] if arrays else np.array([], dtype=float)


def _mask_session_arrays(
    session: dict,
    keys: Sequence[str],
    *,
    domain_filter: str,
    beam_state_filter: str,
    filter_column_keys: Sequence[str],
    quiet_threshold: float | None = None,
) -> list[np.ndarray]:
    """Apply the same sample mask to each *keys* column so they stay aligned."""
    missing = [key for key in keys if key not in session]
    if missing:
        return []
    first = np.asarray(session[keys[0]], dtype=float)
    if first.size == 0:
        return [np.array([], dtype=float) for _ in keys]
    mask = filter_mask_from_columns(
        session,
        filter_column_keys,
        domain_filter,
        beam_state_filter,
    )
    mask = np.asarray(mask, dtype=bool)
    if mask.size != first.size:
        aligned = np.ones(first.size, dtype=bool)
        n = min(mask.size, first.size)
        aligned[:n] = mask[:n]
        mask = aligned
    if beam_state_filter == FILTER_BEAM_OFF:
        from .fft_data import _current_quiet_mask

        thresh = 10.0 if quiet_threshold is None else quiet_threshold
        mask = mask & _current_quiet_mask(first, thresh)
    out: list[np.ndarray] = []
    for key in keys:
        arr = np.asarray(session[key], dtype=float)
        if arr.size == mask.size:
            out.append(arr[mask])
            continue
        n = min(arr.size, mask.size)
        out.append(arr[:n][mask[:n]])
    return out


def _filter_cache_component(config: AudioPlayerConfig) -> tuple[str, ...]:
    """Columns that affect extraction; omit when domain filter ignores them."""
    if config.domain_filter == FILTER_ALL:
        return ()
    return tuple(sorted(config.column_keys))


def processing_cache_key(
    session_id: str,
    config: AudioPlayerConfig,
    channel_id: str,
) -> CacheKey:
    return (
        session_id,
        config.metric_id,
        channel_id,
        config.domain_filter,
        config.beam_state_filter,
        config.beam_subtract,
        _filter_cache_component(config),
    )


def _sanitize_envelope(
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_min = np.nan_to_num(y_min, nan=0.0)
    y_max = np.nan_to_num(y_max, nan=0.0)
    lo = np.minimum(y_min, y_max)
    hi = np.maximum(y_min, y_max)
    return lo, hi


def _envelope_polygon(
    t: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> np.ndarray:
    """Closed polygon tracing max forward then min backward."""
    n = len(t)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    upper = np.column_stack([t, y_max])
    lower = np.column_stack([t[::-1], y_min[::-1]])
    return np.vstack([upper, lower]).astype(np.float32)


def _peak_hold_line(t: np.ndarray, y_min: np.ndarray, y_max: np.ndarray) -> np.ndarray:
    """Polyline through the larger-magnitude envelope edge of each bin."""
    y = np.where(np.abs(y_max) >= np.abs(y_min), y_max, y_min)
    return np.column_stack((t, y)).astype(np.float32)


def _envelope_from_signal(
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x_ms, y_min, y_max = compress_minmax(signal, _ENVELOPE_BINS)
    t = x_ms / 1000.0
    y_min, y_max = _sanitize_envelope(y_min, y_max)
    row_min = float(np.min(y_min))
    row_max = float(np.max(y_max))
    if row_max <= row_min:
        row_max = row_min + 1.0
    pad = max((row_max - row_min) * 0.08, 0.05)
    y_lo = row_min - pad
    y_hi = row_max + pad
    envelope_poly = _envelope_polygon(t, y_min, y_max)
    return envelope_poly, _peak_hold_line(t, y_min, y_max), y_lo, y_hi


def _build_render_channel(
    session: dict,
    config: AudioPlayerConfig,
    channel_id: str,
) -> WaveformRenderChannel | None:
    metric = config.metric
    if metric is None:
        return None

    filter_keys = list(config.column_keys)
    has_ic3 = session_has_ic3(session)
    beam_subtract_enabled = (
        config.beam_subtract
        and config.metric_id == METRIC_IC_CURRENT
        and has_ic3
    )

    subtract = (
        beam_subtract_enabled
        and channel_id in _BEAM_SUBTRACT_CHANNELS
        and "ic3" in session
    )
    if subtract:
        pair = _mask_session_arrays(
            session,
            (channel_id, "ic3"),
            domain_filter=config.domain_filter,
            beam_state_filter=config.beam_state_filter,
            filter_column_keys=filter_keys,
            quiet_threshold=metric.beam_off_quiet_threshold,
        )
        if not pair or pair[0].size == 0:
            return None
        processed = beam_subtract(pair[0], pair[1])
    else:
        raw = extract_audio_signal(
            session,
            channel_id,
            domain_filter=config.domain_filter,
            beam_state_filter=config.beam_state_filter,
            filter_column_keys=filter_keys,
            beam_off_quiet_threshold=metric.beam_off_quiet_threshold,
        )
        if raw.size == 0:
            return None
        processed = raw

    signal = normalize(processed)
    envelope_poly, line_pos, y_lo, y_hi = _envelope_from_signal(signal)
    label = _channel_label(channel_id, beam_subtracted=subtract)
    return WaveformRenderChannel(
        label=label,
        channel_id=channel_id,
        signal=signal,
        color=channel_color(channel_id, beam_subtracted=subtract),
        envelope_poly=envelope_poly,
        line_pos=line_pos,
        y_lo=y_lo,
        y_hi=y_hi,
    )


def prepare_waveform_render(
    session_id: str,
    session: dict,
    config: AudioPlayerConfig,
    cache: dict[CacheKey, WaveformRenderChannel],
) -> list[WaveformRenderChannel]:
    """Build render-ready channels off the UI thread, with per-channel caching."""
    if not config.channels:
        return []

    channels: list[WaveformRenderChannel] = []
    for channel_id in config.channels:
        key = processing_cache_key(session_id, config, channel_id)
        cached = cache.get(key)
        if cached is not None:
            channels.append(cached)
            continue
        built = _build_render_channel(session, config, channel_id)
        if built is None:
            continue
        cache[key] = built
        channels.append(built)
    return channels


def build_playback_channels(
    session: dict,
    config: AudioPlayerConfig,
) -> list[PlaybackChannel]:
    """Build normalized playback channels for the active session."""
    session_id = str(session.get("session_id", ""))
    return list(
        prepare_waveform_render(session_id, session, config, cache={}),
    )


def session_has_ic3(session: dict) -> bool:
    ic3 = session.get("ic3")
    return ic3 is not None and len(ic3) > 0


def format_play_time(seconds: float) -> str:
    """Format a playback position as ``m:ss.t``."""
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(seconds, 60.0)
    return f"{int(minutes)}:{sec:04.1f}"


def live_spectrum(
    signal: np.ndarray,
    time_s: float,
    window_ms: float,
    *,
    fs: float = FS_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed magnitude spectrum around *time_s*, in dB relative to peak."""
    nwin = max(16, int(round(float(window_ms) * 0.001 * fs)))
    freqs = np.fft.rfftfreq(nwin, d=1.0 / fs)
    sig = np.asarray(signal, dtype=np.float64)
    chunk = np.zeros(nwin, dtype=np.float64)
    n = len(sig)
    if n > 0:
        center = int(round(float(time_s) * fs))
        lo = center - nwin // 2
        src_lo = max(0, lo)
        src_hi = min(n, lo + nwin)
        dst_lo = src_lo - lo
        if src_hi > src_lo:
            chunk[dst_lo : dst_lo + (src_hi - src_lo)] = sig[src_lo:src_hi]
    chunk *= np.hanning(nwin)
    mag = np.abs(np.fft.rfft(chunk))
    peak = float(np.max(mag)) if mag.size else 0.0
    floor_lin = 10.0 ** (SPECTRUM_DB_FLOOR / 20.0)
    if peak <= 1e-20:
        db = np.full(mag.shape, SPECTRUM_DB_FLOOR, dtype=np.float64)
    else:
        db = 20.0 * np.log10(np.maximum(mag / peak, floor_lin))
    return freqs, db
