"""Shared matplotlib UI for timeslice replay views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

from ..common import (
    DEFAULT_SESSION_COLORS,
    GRID_KW,
    REFLINE_KW,
    SCATTER_ALPHA,
    SCATTER_SIZE,
    set_view_header,
    VIEW_HEADER_SUBPLOT_TOP,
)
from .timeslice_replay_common import (
    DETAIL_MAX_POINTS,
    MS_PER_SLICE,
    TIMELINE_BINS,
    align_beam_zero_to_primary,
    compress_minmax,
    connect_timeline_interaction,
    detail_line_style,
    draw_beam_off_edges,
    draw_detail_layer_markers,
    draw_digital_row,
    draw_timeline_layer_markers,
    plot_beam_on_twin,
    time_axis,
    window_xy,
)


@dataclass(frozen=True)
class TraceSpec:
    """One row in the detail panel."""

    key: str
    label: str
    color: str
    linewidth: float = 0.5
    beam_off_edges: bool = False


@dataclass(frozen=True)
class ScatterSpec:
    """Right-column scatter configuration."""

    mode: Literal["none", "single", "per_trace"]
    x_key: str = ""
    y_key: str = ""
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    # trace_key → (x_array_key, y_array_key); title uses trace label + suffix
    per_trace_xy: dict[str, tuple[str, str]] = field(default_factory=dict)
    per_trace_title_suffix: str = " Position"
    missing_label: str = "No position data"


@dataclass(frozen=True)
class TimesliceReplayConfig:
    """Launch parameters for a timeslice replay viewer."""

    title: str
    no_data_message: str
    traces: tuple[TraceSpec, ...]
    timeline_key: str
    timeline_ylabel: str
    figsize: tuple[float, float] = (22, 10)
    scatter: ScatterSpec = field(default_factory=lambda: ScatterSpec(mode="none"))
    # Single-session: overlay other traces as faint dotted lines (derived IC view).
    peer_overlay: bool = False
    peer_overlay_linestyle: tuple[int, tuple[int, ...]] = (0, (1, 1))
    peer_overlay_alpha: float = 0.55
    peer_overlay_linewidth: float = 0.5


def launch_timeslice_replay(
    config: TimesliceReplayConfig,
    session_data: dict[str, dict],
    base_dir: str,
) -> None:
    """Build and show the interactive timeslice replay figure."""
    if not session_data:
        print(config.no_data_message)
        return

    loaded_ids = list(session_data.keys())
    sess_colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    multi = len(loaded_ids) > 1
    max_n = max(d["n_samples"] for d in session_data.values())
    show_beam = any(d.get("has_beam", False) for d in session_data.values())
    has_digital = any(d.get("digital") for d in session_data.values())

    trace_keys = [t.key for t in config.traces]
    trace_labels = [t.label for t in config.traces]
    trace_colors = [t.color for t in config.traces]
    n_detail = len(config.traces)

    scatter = config.scatter
    show_scatter = scatter.mode != "none"
    n_cols = 2 if show_scatter else 1
    width_ratios = [4, 1] if show_scatter else [1]

    n_rows = n_detail + (1 if has_digital else 0) + 1
    heights = [1] * n_detail
    if has_digital:
        heights.append(0.25)
    heights.append(0.30)

    fig = plt.figure(figsize=config.figsize)
    set_view_header(fig, config.title, loaded_ids, sess_colors, base_dir=base_dir)

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        height_ratios=heights,
        width_ratios=width_ratios,
        hspace=0.18, wspace=0.04,
        top=VIEW_HEADER_SUBPLOT_TOP, bottom=0.06, left=0.03, right=0.99,
    )

    ax_detail = [fig.add_subplot(gs[0, 0])]
    for i in range(1, n_detail):
        ax_detail.append(fig.add_subplot(gs[i, 0], sharex=ax_detail[0]))

    ax_digital: plt.Axes | None = None
    digital_row = n_detail
    if has_digital:
        ax_digital = fig.add_subplot(gs[digital_row, 0], sharex=ax_detail[0])
        digital_row += 1

    ax_timeline = fig.add_subplot(gs[digital_row, :])

    ax_scatter_list: list[plt.Axes | None] = [None] * n_detail
    ax_scatter_single: plt.Axes | None = None
    if scatter.mode == "single":
        ax_scatter_single = fig.add_subplot(gs[0:n_detail, 1])
    elif scatter.mode == "per_trace":
        for i in range(n_detail):
            ax_scatter_list[i] = fig.add_subplot(gs[i, 1])

    for ax in ax_detail:
        plt.setp(ax.get_xticklabels(), visible=False)
    if ax_digital is not None:
        plt.setp(ax_digital.get_xticklabels(), visible=False)

    for si, (sid, data) in enumerate(session_data.items()):
        signal = data.get(config.timeline_key)
        if signal is None:
            continue
        x, y_min, y_max = compress_minmax(signal, TIMELINE_BINS)
        short = trace_labels[0] if trace_labels else config.timeline_key
        label = f"{short} — {sid}" if multi else short
        ax_timeline.fill_between(
            x, y_min, y_max, alpha=0.45,
            color=sess_colors[si], label=label,
        )

    max_t = max_n * MS_PER_SLICE
    ax_timeline.set_xlim(0, max_t)
    ax_timeline.set_ylabel(config.timeline_ylabel, fontsize=8)
    time_axis(ax_timeline, 0, max_t)
    ax_timeline.grid(**GRID_KW, which="major")
    ax_timeline.grid(which="minor", color="#e0e0e0", linewidth=0.3)
    ax_timeline.tick_params(labelsize=8)

    for data in session_data.values():
        draw_timeline_layer_markers(ax_timeline, data["layer_boundaries"])

    all_energies = set()
    for data in session_data.values():
        all_energies.update(np.unique(data["energy"]))
    if all_energies:
        e_min, e_max = min(all_energies), max(all_energies)
    else:
        e_min, e_max = 0, 1
    norm = mcolors.Normalize(vmin=e_min, vmax=e_max)

    beam_twins: list[plt.Axes | None] = [None] * n_detail

    def _draw_detail(xmin_t: float, xmax_t: float) -> None:
        lo = max(0, int(xmin_t / MS_PER_SLICE))
        hi = min(max_n, int(xmax_t / MS_PER_SLICE))
        if hi <= lo:
            return
        lo_t = lo * MS_PER_SLICE
        hi_t = hi * MS_PER_SLICE

        for t_idx, spec in enumerate(config.traces):
            ax = ax_detail[t_idx]
            ax.clear()

            if beam_twins[t_idx] is not None:
                beam_twins[t_idx].remove()
                beam_twins[t_idx] = None

            for si, (sid, data) in enumerate(session_data.items()):
                sig = data.get(spec.key)
                if sig is None:
                    continue
                xy = window_xy(sig, lo, hi)
                if xy is None:
                    continue
                plot_x, plot_y = xy

                line_color = sess_colors[si] if multi else spec.color
                primary_label = spec.label if (t_idx == 0 and not multi) else None
                ax.plot(
                    plot_x, plot_y,
                    color=line_color,
                    linewidth=spec.linewidth,
                    label=primary_label,
                    zorder=3,
                    **detail_line_style(len(plot_y)),
                )

                if config.peer_overlay and not multi:
                    for other_idx, other_spec in enumerate(config.traces):
                        if other_spec.key == spec.key:
                            continue
                        other_sig = data.get(other_spec.key)
                        if other_sig is None:
                            continue
                        oxy = window_xy(other_sig, lo, hi)
                        if oxy is None:
                            continue
                        opx, opy = oxy
                        ax.plot(
                            opx, opy,
                            color=trace_colors[other_idx],
                            linewidth=config.peer_overlay_linewidth,
                            linestyle=config.peer_overlay_linestyle,
                            alpha=config.peer_overlay_alpha,
                            zorder=2,
                            label=other_spec.label if t_idx == 0 else None,
                            **detail_line_style(len(opy)),
                        )

                if spec.beam_off_edges:
                    edges = data.get("beam_off_edges", {}).get(spec.key)
                    draw_beam_off_edges(ax, edges, lo, hi)

            if show_beam:
                ax_b = ax.twinx()
                beam_twins[t_idx] = ax_b
                for si, (sid, data) in enumerate(session_data.items()):
                    beam = data.get("beam")
                    if beam is None:
                        continue
                    beam_color = sess_colors[si] if multi else spec.color
                    beam_label = f"Beam I ({sid})" if multi else "Beam I"
                    plot_beam_on_twin(
                        ax, ax_b, beam, lo, hi,
                        color=beam_color,
                        label=beam_label if t_idx == 0 else None,
                    )
                ax_b.set_ylabel("Beam I", fontsize=8)
                ax_b.tick_params(axis="y", labelsize=7)
                align_beam_zero_to_primary(ax, ax_b)

            for data in session_data.values():
                draw_detail_layer_markers(
                    ax, data["layer_boundaries"], lo_t, hi_t,
                    annotate=(t_idx == 0),
                )

            ax.set_xlim(lo_t, hi_t)
            ax.set_ylabel(spec.label, fontsize=9)
            ax.grid(**GRID_KW, which="major")
            ax.grid(which="minor", color="#e0e0e0", linewidth=0.3)

        if ax_digital is not None:
            draw_digital_row(
                ax_digital, session_data, lo, hi, lo_t, hi_t, multi=multi,
            )

        for a in ax_detail:
            plt.setp(a.get_xticklabels(), visible=False)
        last_detail = ax_digital if ax_digital is not None else ax_detail[-1]
        time_axis(last_detail, lo_t, hi_t)
        plt.setp(last_detail.get_xticklabels(), visible=True)
        for a in ax_detail:
            a.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        if ax_digital is not None:
            ax_digital.xaxis.set_minor_locator(mticker.AutoMinorLocator())

        handles, labels = ax_detail[0].get_legend_handles_labels()
        if show_beam and beam_twins[0] is not None:
            bh, bl = beam_twins[0].get_legend_handles_labels()
            handles += bh
            labels += bl
        if handles:
            ax_detail[0].legend(handles, labels, loc="upper right", fontsize=8)

        if scatter.mode == "single" and ax_scatter_single is not None:
            ax_scatter_single.clear()
            for si, (sid, data) in enumerate(session_data.items()):
                n = data["n_samples"]
                a_lo = min(lo, n)
                a_hi = min(hi, n)
                if a_hi <= a_lo:
                    continue
                w_energy = data["energy"][a_lo:a_hi]
                wx = data[scatter.x_key][a_lo:a_hi]
                wy = data[scatter.y_key][a_lo:a_hi]
                w_len = a_hi - a_lo
                if w_len > DETAIL_MAX_POINTS:
                    step = max(1, w_len // DETAIL_MAX_POINTS)
                    sl = slice(None, None, step)
                    w_energy, wx, wy = w_energy[sl], wx[sl], wy[sl]
                valid = np.isfinite(wx) & np.isfinite(wy)
                if not valid.any():
                    continue
                ax_scatter_single.scatter(
                    wx[valid], wy[valid],
                    c=w_energy[valid], cmap="viridis", norm=norm,
                    alpha=SCATTER_ALPHA, s=SCATTER_SIZE, edgecolors="none",
                )
            ax_scatter_single.axhline(y=0, **REFLINE_KW)
            ax_scatter_single.axvline(x=0, **REFLINE_KW)
            ax_scatter_single.grid(**GRID_KW)
            ax_scatter_single.set_aspect("equal", adjustable="datalim")
            ax_scatter_single.tick_params(labelsize=7)
            ax_scatter_single.set_title(scatter.title, fontsize=9)
            ax_scatter_single.set_xlabel(scatter.xlabel, fontsize=8)
            ax_scatter_single.set_ylabel(scatter.ylabel, fontsize=8)

        elif scatter.mode == "per_trace":
            for t_idx, spec in enumerate(config.traces):
                ax_sc = ax_scatter_list[t_idx]
                if ax_sc is None:
                    continue
                ax_sc.clear()
                xy_keys = scatter.per_trace_xy.get(spec.key)
                if xy_keys is None:
                    ax_sc.text(
                        0.5, 0.5, scatter.missing_label,
                        transform=ax_sc.transAxes, ha="center", va="center",
                        fontsize=9, color="gray", alpha=0.6,
                    )
                else:
                    xk, yk = xy_keys
                    for si, (sid, data) in enumerate(session_data.items()):
                        if not data.get("has_positions", True):
                            continue
                        n = data["n_samples"]
                        a_lo = min(lo, n)
                        a_hi = min(hi, n)
                        if a_hi <= a_lo:
                            continue
                        w_energy = data["energy"][a_lo:a_hi]
                        wx = data[xk][a_lo:a_hi]
                        wy = data[yk][a_lo:a_hi]
                        w_len = a_hi - a_lo
                        if w_len > DETAIL_MAX_POINTS:
                            step = max(1, w_len // DETAIL_MAX_POINTS)
                            sl = slice(None, None, step)
                            w_energy, wx, wy = w_energy[sl], wx[sl], wy[sl]
                        valid = np.isfinite(wx) & np.isfinite(wy)
                        if not valid.any():
                            continue
                        ax_sc.scatter(
                            wx[valid], wy[valid],
                            c=w_energy[valid], cmap="viridis", norm=norm,
                            alpha=SCATTER_ALPHA, s=SCATTER_SIZE, edgecolors="none",
                        )
                ax_sc.axhline(y=0, **REFLINE_KW)
                ax_sc.axvline(x=0, **REFLINE_KW)
                ax_sc.grid(**GRID_KW)
                ax_sc.set_aspect("equal", adjustable="datalim")
                ax_sc.tick_params(labelsize=7)
                ax_sc.set_title(
                    f"{spec.label}{scatter.per_trace_title_suffix}", fontsize=9,
                )
                if t_idx == n_detail - 1:
                    ax_sc.set_xlabel("X (mm)", fontsize=8)

        fig.canvas.draw_idle()

    connect_timeline_interaction(fig, ax_timeline, _draw_detail, max_t)
    plt.show()
