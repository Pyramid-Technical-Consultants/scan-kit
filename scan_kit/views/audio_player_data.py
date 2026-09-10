"""Timeslice signal preparation for the IC audio player."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..common.data_filter import (
    FILTER_ALL,
    FILTER_BEAM_BOTH,
    filter_mask_from_columns,
)
from ..data.timeline_channels import TIMELINE_CHANNEL_BY_KEY
from .fft_catalog import CHANNEL_BY_ID, FftConfig, METRIC_IC_CURRENT
from .fft_data import FS_HZ, extract_fft_traces
from .timeslice_replay_common import compress_minmax

_ENVELOPE_BINS = 2000

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
    """Playback channel with precomputed vispy envelope geometry."""

    mesh_pos: np.ndarray
    mesh_faces: np.ndarray
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


def _specsub_pass(
    signal: np.ndarray,
    ref_ac: np.ndarray,
    nperseg: int,
    noverlap: int,
) -> np.ndarray:
    from scipy.signal import istft, stft

    n = len(signal)
    _, _, zt = stft(signal, fs=FS_HZ, nperseg=nperseg, noverlap=noverlap)
    _, _, zr = stft(ref_ac, fs=FS_HZ, nperseg=nperseg, noverlap=noverlap)

    mag_t = np.abs(zt)
    mag_r = np.abs(zr)
    e_t = np.sum(mag_t ** 2, axis=0, keepdims=True)
    e_r = np.sum(mag_r ** 2, axis=0, keepdims=True)
    alpha_frame = np.sqrt(e_t / (e_r + 1e-20))

    beam_estimate = _SPECSUB_ALPHA * alpha_frame * mag_r
    clean_mag = np.maximum(mag_t - beam_estimate, _SPECSUB_BETA * mag_t)
    z_clean = clean_mag * np.exp(1j * np.angle(zt))
    _, out = istft(z_clean, fs=FS_HZ, nperseg=nperseg, noverlap=noverlap)
    return out[:n]


def beam_subtract(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remove beam-correlated content from *target* via spectral subtraction."""
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

    if beam_state_filter == FILTER_BEAM_BOTH:
        sig = np.asarray(session[channel_id], dtype=float)
        domain_mask = filter_mask_from_columns(
            session,
            filter_column_keys,
            domain_filter,
            FILTER_BEAM_BOTH,
        )
        if domain_mask is None:
            return sig
        mask = np.asarray(domain_mask, dtype=bool)
        return sig[mask] if np.any(mask) else np.array([], dtype=float)

    traces = extract_fft_traces(
        session,
        channel_id,
        domain_filter=domain_filter,
        beam_state_filter=beam_state_filter,
        filter_column_keys=filter_column_keys,
        beam_off_quiet_threshold=beam_off_quiet_threshold,
    )
    if not traces:
        return np.array([], dtype=float)
    if len(traces) == 1:
        return traces[0][0]
    parts = [trace[0] for trace in traces if trace[0].size]
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def _filter_cache_component(config: AudioPlayerConfig) -> tuple[str, ...]:
    """Columns that affect extraction; omit when filters ignore channel selection."""
    if (
        config.domain_filter == FILTER_ALL
        and config.beam_state_filter == FILTER_BEAM_BOTH
    ):
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


def _envelope_mesh(
    t: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(t)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)

    verts = np.empty((n * 2, 2), dtype=np.float32)
    verts[0::2, 0] = t
    verts[0::2, 1] = y_min
    verts[1::2, 0] = t
    verts[1::2, 1] = y_max

    faces = np.empty((max(0, n - 1) * 2, 3), dtype=np.uint32)
    for i in range(n - 1):
        lo = i * 2
        hi = lo + 2
        row = i * 2
        faces[row] = (lo, lo + 1, hi + 1)
        faces[row + 1] = (lo, hi + 1, hi)
    return verts, faces


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
    mesh_pos, mesh_faces = _envelope_mesh(t, y_min, y_max)
    return mesh_pos, mesh_faces, y_lo, y_hi


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
    ic3_raw = np.asarray(session["ic3"], dtype=float) if has_ic3 else None

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

    subtract = (
        beam_subtract_enabled
        and channel_id in _BEAM_SUBTRACT_CHANNELS
        and ic3_raw is not None
    )
    if subtract:
        n = min(len(raw), len(ic3_raw))
        processed = beam_subtract(raw[:n], ic3_raw[:n])
    else:
        processed = raw

    signal = normalize(processed)
    mesh_pos, mesh_faces, y_lo, y_hi = _envelope_from_signal(signal)
    label = _channel_label(channel_id, beam_subtracted=subtract)
    return WaveformRenderChannel(
        label=label,
        channel_id=channel_id,
        signal=signal,
        color=channel_color(channel_id, beam_subtracted=subtract),
        mesh_pos=mesh_pos,
        mesh_faces=mesh_faces,
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
