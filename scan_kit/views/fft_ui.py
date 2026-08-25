"""Matplotlib renderer for the FFT Explorer viewer."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

from ..common import (
    DEFAULT_SESSION_COLORS,
    GRID_KW,
    VIEW_HEADER_SUBPLOT_TOP,
    set_view_header,
)
from ..common.data_filter import FILTER_BEAM_BOTH
from .fft_catalog import FftConfig
from .fft_data import (
    FREQ_MAX_HZ,
    FREQ_MIN_HZ,
    FS_HZ,
    OVERLAP_FRACTION,
    SEGMENT_LENGTH,
    extract_fft_traces,
    find_peak_indices,
    welch_psd,
)

CHANNEL_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

_LABEL_MIN_X_GAP_PX = 50
_LABEL_OFFSET_LOW = 8
_LABEL_OFFSET_HIGH = 22


def _annotate_peaks(ax, freqs: np.ndarray, psd: np.ndarray, color: str) -> None:
    indices = find_peak_indices(psd)
    if len(indices) == 0:
        return

    placed: list[tuple[float, float]] = []
    transform = ax.transData

    for idx in indices:
        freq = freqs[idx]
        amp = psd[idx]
        x_disp = transform.transform((freq, amp))[0]

        y_off = _LABEL_OFFSET_LOW
        for prev_x, prev_off in placed:
            if abs(x_disp - prev_x) < _LABEL_MIN_X_GAP_PX:
                y_off = (
                    _LABEL_OFFSET_HIGH
                    if prev_off == _LABEL_OFFSET_LOW
                    else _LABEL_OFFSET_LOW
                )
                break

        ax.annotate(
            f"{freq:.1f} Hz",
            xy=(freq, amp),
            xytext=(0, y_off),
            textcoords="offset points",
            fontsize=7,
            color=color,
            fontweight="bold",
            ha="center",
            va="bottom",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
        )
        placed.append((x_disp, y_off))


def _style_fft_ax(ax: plt.Axes, *, psd_unit: str) -> None:
    ax.set_xlim(FREQ_MIN_HZ, FREQ_MAX_HZ)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.tick_params(which="minor", length=3)
    ax.set_ylabel(f"PSD ({psd_unit}²/Hz)", fontsize=9)
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.grid(**GRID_KW)
    ax.grid(which="minor", color="#e0e0e0", linewidth=0.3, alpha=0.5)


def _linestyle_legend(ax: plt.Axes, *, beam_state_filter: str, loc: str = "upper right") -> None:
    handles = [
        Line2D([0], [0], color="0.35", linestyle="-", linewidth=0.9, label="Beam on"),
    ]
    if beam_state_filter == FILTER_BEAM_BOTH:
        handles.append(
            Line2D([0], [0], color="0.35", linestyle="--", linewidth=0.9, label="Beam off"),
        )
    ax.legend(
        handles=handles,
        loc=loc,
        fontsize=8,
        frameon=True,
        framealpha=0.9,
    )


def render_fft(
    fig: plt.Figure,
    config: FftConfig,
    session_data: dict[str, dict],
    base_dir: str,
) -> None:
    """Clear *fig* and draw line FFT spectra for *config*."""
    fig.clear()
    if not session_data:
        fig.text(0.5, 0.5, "No timeslice data loaded", ha="center", va="center")
        fig.canvas.draw_idle()
        return

    channel_defs = list(config.channel_defs)
    filter_column_keys = list(config.column_keys)
    metric = config.metric
    quiet_threshold = metric.beam_off_quiet_threshold if metric is not None else None
    if not channel_defs:
        fig.text(0.5, 0.5, "No signals selected", ha="center", va="center")
        fig.canvas.draw_idle()
        return

    loaded_ids = list(session_data.keys())
    sess_colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    multi = len(loaded_ids) > 1
    n_channels = len(channel_defs)

    set_view_header(fig, config.title, loaded_ids, sess_colors, base_dir=base_dir)
    gs = fig.add_gridspec(
        1,
        n_channels,
        left=0.05,
        right=0.97,
        top=VIEW_HEADER_SUBPLOT_TOP,
        bottom=0.08,
        wspace=0.18,
    )

    line_axes: list[plt.Axes] = []
    for idx in range(n_channels):
        ax = fig.add_subplot(gs[0, idx])
        line_axes.append(ax)
    for idx in range(1, n_channels):
        line_axes[idx].sharex(line_axes[0])
        line_axes[idx].sharey(line_axes[0])

    for ch_idx, channel in enumerate(channel_defs):
        ax = line_axes[ch_idx]
        for si, (sid, data) in enumerate(session_data.items()):
            color = sess_colors[si] if multi else CHANNEL_COLORS[ch_idx % len(CHANNEL_COLORS)]
            traces = extract_fft_traces(
                data,
                channel.id,
                domain_filter=config.domain_filter,
                beam_state_filter=config.beam_state_filter,
                filter_column_keys=filter_column_keys,
                beam_off_quiet_threshold=quiet_threshold,
            )
            for signal, ls in traces:
                if signal.size == 0:
                    continue
                freqs_hz, psd = welch_psd(
                    signal, FS_HZ, SEGMENT_LENGTH, OVERLAP_FRACTION,
                )
                if freqs_hz is None:
                    continue
                band = (freqs_hz >= FREQ_MIN_HZ) & (freqs_hz <= FREQ_MAX_HZ)
                ax.semilogy(
                    freqs_hz[band],
                    psd[band],
                    linewidth=0.6,
                    alpha=0.85,
                    color=color,
                    linestyle=ls,
                )
                if ls == "-" and config.annotate_peaks:
                    _annotate_peaks(ax, freqs_hz[band], psd[band], color)

        _style_fft_ax(ax, psd_unit=channel.psd_unit)
        ax.set_title(channel.label, fontsize=11, fontweight="bold")

    _linestyle_legend(line_axes[-1], beam_state_filter=config.beam_state_filter)
    fig.canvas.draw_idle()
