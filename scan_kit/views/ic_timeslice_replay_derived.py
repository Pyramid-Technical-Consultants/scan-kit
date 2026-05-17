"""IC timeslice replay using IC current derived from the scan-total dose.

Mirrors :mod:`ic_timeslice_replay` but, instead of reading the IC primary /
quad current columns, derives a per-slice IC current by differentiating the
monotonically-accumulating *scan total dose* columns
(``r_ic1_scan_total_dose``, ``r_ic2_scan_total_dose``, ``ic3_dose_total``).

The derivative units are *dose-units per timeslice* (with one timeslice = 1 ms
by convention).  Magnitudes will not match the canonical nA-scaled IC current
columns, but the temporal shape — including beam-on/off transitions — should
agree closely with the raw current.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.widgets import SpanSelector

from ..common import (
    C_BEAM_CURRENT,
    C_ENERGY,
    C_IC1_SCAN_TOTAL_DOSE,
    C_IC2_SCAN_TOTAL_DOSE,
    C_IC3_SCAN_TOTAL_DOSE,
    C_IC1_X_POS_RAW,
    C_IC1_Y_POS_RAW,
    C_IC2_X_POS_RAW,
    C_IC2_Y_POS_RAW,
    C_LAYER_ID,
    resolve_concept_column,
    DEFAULT_SESSION_COLORS,
    SUPTITLE_KW,
    GRID_KW,
    REFLINE_KW,
    SCATTER_ALPHA,
    SCATTER_SIZE,
)
from ..common.session_source import (
    load_session_csv,
    load_session_timeslice_device_units,
    resolve_session_source,
)
from ..common.schema import (
    POSITION_KEY_G2_RAW,
    POSITION_KEY_G3_RAW,
)
from ..common import transform
from .beam_off_rampdown import detect_beam_off_edges

TIMELINE_BINS = 4000
DETAIL_MAX_POINTS = 80_000
MS_PER_SLICE = 1.0


def _time_axis(ax: plt.Axes, lo_ms: float, hi_ms: float) -> None:
    """Configure *ax* X-axis to display time with adaptive units and dense ticks."""
    span_ms = hi_ms - lo_ms
    if span_ms <= 0:
        return

    if span_ms > 10_000:
        scale = 1e-3
        unit = "s"
    else:
        scale = 1.0
        unit = "ms"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v * scale:.6g}"
    ))
    ax.set_xlabel(f"Time ({unit})")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(axis="x", which="minor", length=3)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _resolve_col(columns, concept: str) -> str | None:
    """Resolve a concept to the actual column name present in *columns*."""
    return resolve_concept_column(columns, concept)


def _derive_current_from_dose(dose: np.ndarray) -> np.ndarray:
    """Per-slice derivative of a monotonically-accumulating dose column.

    Returns an array of the same length as *dose*: the first sample is set to
    0 (no prior reference), and remaining samples hold ``dose[i] - dose[i-1]``
    divided by ``MS_PER_SLICE`` so the output reads as *dose-units per ms*.

    Negative deltas (which occasionally occur due to floating-point quantization
    on the streamed dose register) are clipped to zero — they are not physical
    decreases in accumulated dose.
    """
    if len(dose) == 0:
        return np.empty(0, dtype=float)
    arr = np.asarray(dose, dtype=float)
    deriv = np.empty_like(arr)
    deriv[0] = 0.0
    if len(arr) > 1:
        deriv[1:] = (arr[1:] - arr[:-1]) / MS_PER_SLICE
    deriv[~np.isfinite(deriv)] = 0.0
    deriv[deriv < 0] = 0.0
    return deriv


def _load_session_timeline(session_id: str, base_dir: str) -> dict | None:
    """Load and concatenate all timeslice frames into a unified timeline.

    The IC "current" arrays returned are derived by differentiating the
    scan-total dose columns rather than read from the IC current columns.
    """
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    input_map = load_session_csv(src, "input_map.csv")
    if input_map is None:
        return None

    col_layer = _resolve_col(input_map.columns, C_LAYER_ID)
    col_energy = _resolve_col(input_map.columns, C_ENERGY)
    if col_layer is None or col_energy is None:
        return None

    energy_by_layer = input_map.groupby(col_layer)[col_energy].first().to_dict()

    frames = load_session_timeslice_device_units(src)
    if not frames:
        return None

    df0 = frames[0]
    ts_layer = _resolve_col(df0.columns, C_LAYER_ID)
    ts_dose1 = _resolve_col(df0.columns, C_IC1_SCAN_TOTAL_DOSE)
    ts_dose2 = _resolve_col(df0.columns, C_IC2_SCAN_TOTAL_DOSE)
    if not all([ts_layer, ts_dose1, ts_dose2]):
        return None

    ts_dose3 = _resolve_col(df0.columns, C_IC3_SCAN_TOTAL_DOSE)
    has_ic3 = ts_dose3 is not None

    ts_beam = _resolve_col(df0.columns, C_BEAM_CURRENT)
    has_beam = ts_beam is not None

    pos_cols: dict[str, str] = {}
    for pos_key in (POSITION_KEY_G3_RAW, POSITION_KEY_G2_RAW):
        for concept, label in [
            (C_IC1_X_POS_RAW, "ic1_x"), (C_IC1_Y_POS_RAW, "ic1_y"),
            (C_IC2_X_POS_RAW, "ic2_x"), (C_IC2_Y_POS_RAW, "ic2_y"),
        ]:
            resolved = resolve_concept_column(df0.columns, concept, position_key=pos_key)
            if resolved and label not in pos_cols:
                pos_cols[label] = resolved
        if len(pos_cols) == 4:
            break
    has_positions = len(pos_cols) == 4

    _DIGITAL_DEFS: list[tuple[str, str, str]] = [
        ("rci_in_trigger", "rci_in_trigger", "RCI Trigger"),
        ("r_beamOk", "r_beamOk", "Beam OK"),
    ]
    digital_cols: list[tuple[str, str]] = []
    for raw_col, _canonical, label in _DIGITAL_DEFS:
        if raw_col in df0.columns:
            digital_cols.append((raw_col, label))

    ic1_parts: list[np.ndarray] = []
    ic2_parts: list[np.ndarray] = []
    ic3_parts: list[np.ndarray] = []
    beam_parts: list[np.ndarray] = []
    digital_parts: dict[str, list[np.ndarray]] = {col: [] for col, _ in digital_cols}
    pos_parts: dict[str, list[np.ndarray]] = {k: [] for k in ("ic1_x", "ic1_y", "ic2_x", "ic2_y")}
    energy_parts: list[np.ndarray] = []
    layer_boundaries: list[tuple[int, float]] = []
    edge_indices: dict[str, list[int]] = {"ic1": [], "ic2": [], "ic3": []}
    offset = 0

    for df in frames:
        n = len(df)
        layer_id = df[ts_layer].iloc[0]
        energy = energy_by_layer.get(layer_id, 0.0)

        # Scan-total dose accumulates across the whole session, but each frame
        # is a per-layer CSV — so differentiate within the frame to avoid a
        # spurious spike at the start of the next layer caused by the first
        # sample being unreferenced.
        ic1_vals = _derive_current_from_dose(df[ts_dose1].values.astype(float))
        ic2_vals = _derive_current_from_dose(df[ts_dose2].values.astype(float))
        ic1_parts.append(ic1_vals)
        ic2_parts.append(ic2_vals)

        for key, vals in [("ic1", ic1_vals), ("ic2", ic2_vals)]:
            edges = detect_beam_off_edges(vals)
            edge_indices[key].extend((edges + offset).tolist())

        if has_ic3:
            ic3_vals = _derive_current_from_dose(df[ts_dose3].values.astype(float))
            ic3_parts.append(ic3_vals)
            edges = detect_beam_off_edges(ic3_vals)
            edge_indices["ic3"].extend((edges + offset).tolist())

        if has_beam:
            beam_parts.append(df[ts_beam].values.astype(float))
        for col, _ in digital_cols:
            if col in df.columns:
                digital_parts[col].append(df[col].values.astype(float))
            else:
                digital_parts[col].append(np.zeros(n))
        if has_positions:
            for label, col in pos_cols.items():
                pos_parts[label].append(df[col].values.astype(float))
        energy_parts.append(np.full(n, energy))
        layer_boundaries.append((offset, energy))
        offset += n

    if offset == 0:
        return None

    digital_signals: dict[str, tuple[np.ndarray, str]] = {}
    for col, label in digital_cols:
        if digital_parts[col]:
            digital_signals[col] = (np.concatenate(digital_parts[col]), label)

    result: dict = {
        "ic1": np.concatenate(ic1_parts),
        "ic2": np.concatenate(ic2_parts),
        "layer_boundaries": layer_boundaries,
        "n_samples": offset,
        "has_ic3": has_ic3,
        "has_beam": has_beam,
        "has_positions": has_positions,
        "energy": np.concatenate(energy_parts),
        "beam_off_edges": {k: np.asarray(v, dtype=int) for k, v in edge_indices.items()},
        "digital": digital_signals,
    }
    if has_ic3:
        result["ic3"] = np.concatenate(ic3_parts)
    if has_beam:
        result["beam"] = np.concatenate(beam_parts)
    if has_positions:
        result["ic1_x"] = transform.remap(np.concatenate(pos_parts["ic1_x"]), *transform.IC1_X_MAP)
        result["ic1_y"] = transform.remap(np.concatenate(pos_parts["ic1_y"]), *transform.IC1_Y_MAP)
        result["ic2_x"] = transform.remap(np.concatenate(pos_parts["ic2_x"]), *transform.IC2_X_MAP)
        result["ic2_y"] = transform.remap(np.concatenate(pos_parts["ic2_y"]), *transform.IC2_Y_MAP)
        _POS_LIMIT = transform.IC_MM_MAX
        for k in ("ic1_x", "ic1_y", "ic2_x", "ic2_y"):
            arr = result[k]
            arr[np.abs(arr) > _POS_LIMIT] = np.nan
    return result


def _compress_minmax(signal: np.ndarray, n_bins: int):
    """Compress signal into a min/max envelope for fast timeline rendering."""
    n = len(signal)
    if n <= n_bins * 2:
        x = np.arange(n, dtype=float) * MS_PER_SLICE
        return x, signal.copy(), signal.copy()

    bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
    starts = bin_edges[:-1]
    x = (starts + bin_edges[1:]) * 0.5 * MS_PER_SLICE

    safe = np.where(np.isnan(signal), np.inf, signal)
    y_min = np.minimum.reduceat(safe, starts)

    safe_max = np.where(np.isnan(signal), -np.inf, signal)
    y_max = np.maximum.reduceat(safe_max, starts)

    return x, y_min, y_max


# ---------------------------------------------------------------------------
# View entry point
# ---------------------------------------------------------------------------

def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Launch the IC timeslice replay viewer using dose-derived IC current."""
    if not session_ids:
        print("No sessions selected")
        return

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        data = _load_session_timeline(sid, base_dir)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        print("No valid timeslice data found for any session")
        return

    loaded_ids = list(session_data.keys())
    sess_colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    multi = len(loaded_ids) > 1
    max_n = max(d["n_samples"] for d in session_data.values())
    show_ic3 = any(d.get("has_ic3", False) for d in session_data.values())
    show_beam = any(d.get("has_beam", False) for d in session_data.values())
    show_pos = any(d.get("has_positions", False) for d in session_data.values())

    ic_keys: list[str] = ["ic1", "ic2"]
    ic_labels: list[str] = ["IC1 dDose/dt", "IC2 dDose/dt"]
    ic_detail_colors: list[str] = ["#1f77b4", "#d62728"]
    if show_ic3:
        ic_keys.append("ic3")
        ic_labels.append("IC3 dDose/dt")
        ic_detail_colors.append("#2ca02c")

    n_detail = len(ic_keys)
    has_digital = any(d.get("digital") for d in session_data.values())

    n_cols = 2 if show_pos else 1
    width_ratios = [4, 1] if show_pos else [1]

    n_rows = n_detail + (1 if has_digital else 0) + 1
    heights = [1] * n_detail
    if has_digital:
        heights.append(0.25)
    heights.append(0.30)

    fig = plt.figure(figsize=(22 if show_pos else 18, 10))
    fig.suptitle("IC Timeslice Replay — Current Derived from Scan-Total Dose", **SUPTITLE_KW)

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        height_ratios=heights,
        width_ratios=width_ratios,
        hspace=0.18, wspace=0.04,
        top=0.94, bottom=0.06, left=0.03, right=0.99,
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

    ax_scatter: list[plt.Axes | None] = [None] * n_detail
    if show_pos:
        for i in range(n_detail):
            ax_scatter[i] = fig.add_subplot(gs[i, 1])

    for ax in ax_detail:
        plt.setp(ax.get_xticklabels(), visible=False)
    if ax_digital is not None:
        plt.setp(ax_digital.get_xticklabels(), visible=False)

    for si, (sid, data) in enumerate(session_data.items()):
        x, y_min, y_max = _compress_minmax(data["ic1"], TIMELINE_BINS)
        label = f"IC1 dDose/dt — {sid}" if multi else "IC1 dDose/dt"
        ax_timeline.fill_between(
            x, y_min, y_max, alpha=0.45,
            color=sess_colors[si], label=label,
        )

    max_t = max_n * MS_PER_SLICE
    ax_timeline.set_xlim(0, max_t)
    ax_timeline.set_ylabel("dDose/dt", fontsize=8)
    _time_axis(ax_timeline, 0, max_t)
    ax_timeline.grid(**GRID_KW, which="major")
    ax_timeline.grid(which="minor", color="#e0e0e0", linewidth=0.3)
    ax_timeline.tick_params(labelsize=8)
    if multi:
        ax_timeline.legend(loc="upper right", fontsize=7, ncol=len(loaded_ids))

    for sid_data in session_data.values():
        for offset, _energy in sid_data["layer_boundaries"]:
            if offset > 0:
                ax_timeline.axvline(offset * MS_PER_SLICE, color="gray", linewidth=0.3, alpha=0.25)

    all_energies = set()
    for data in session_data.values():
        all_energies.update(np.unique(data["energy"]))
    if all_energies:
        e_min, e_max = min(all_energies), max(all_energies)
    else:
        e_min, e_max = 0, 1
    norm = mcolors.Normalize(vmin=e_min, vmax=e_max)

    initial_end_t = min(max_t, max(2000 * MS_PER_SLICE, max_t / 10))
    _beam_twins: list[plt.Axes | None] = [None] * n_detail

    def _draw_detail(xmin_t: float, xmax_t: float) -> None:
        lo = max(0, int(xmin_t / MS_PER_SLICE))
        hi = min(max_n, int(xmax_t / MS_PER_SLICE))
        if hi <= lo:
            return
        lo_t = lo * MS_PER_SLICE
        hi_t = hi * MS_PER_SLICE

        for ic_idx, (ic, label) in enumerate(zip(ic_keys, ic_labels)):
            ax = ax_detail[ic_idx]
            ax.clear()

            if _beam_twins[ic_idx] is not None:
                _beam_twins[ic_idx].remove()
                _beam_twins[ic_idx] = None

            def _window_xy(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
                n = len(sig)
                a_lo = min(lo, n)
                a_hi = min(hi, n)
                if a_hi <= a_lo:
                    return None
                window = sig[a_lo:a_hi]
                w_len = a_hi - a_lo
                if w_len > DETAIL_MAX_POINTS:
                    step = max(1, w_len // DETAIL_MAX_POINTS)
                    return np.arange(a_lo, a_hi, step) * MS_PER_SLICE, window[::step]
                return np.arange(a_lo, a_hi) * MS_PER_SLICE, window

            for si, (sid, data) in enumerate(session_data.items()):
                sig = data.get(ic)
                if sig is None:
                    continue
                xy = _window_xy(sig)
                if xy is None:
                    continue
                plot_x, plot_y = xy

                line_color = sess_colors[si] if multi else ic_detail_colors[ic_idx]
                primary_label: str | None = None
                if ic_idx == 0:
                    primary_label = sid if multi else ic_labels[ic_idx]
                ax.plot(
                    plot_x, plot_y,
                    color=line_color, linewidth=0.6,
                    label=primary_label,
                    zorder=3,
                )

                # Single-session mode: overlay the other ICs in their own
                # colors as faint dotted lines so each panel can be compared
                # against its peers without selecting another session.
                if not multi:
                    for other_idx, other_ic in enumerate(ic_keys):
                        if other_idx == ic_idx:
                            continue
                        other_sig = data.get(other_ic)
                        if other_sig is None:
                            continue
                        oxy = _window_xy(other_sig)
                        if oxy is None:
                            continue
                        opx, opy = oxy
                        ax.plot(
                            opx, opy,
                            color=ic_detail_colors[other_idx],
                            linewidth=0.5,
                            linestyle=(0, (1, 1)),
                            alpha=0.55,
                            zorder=2,
                            label=ic_labels[other_idx] if ic_idx == 0 else None,
                        )

                a_lo = min(lo, len(sig))
                a_hi = min(hi, len(sig))
                edges = data.get("beam_off_edges", {}).get(ic)
                if edges is not None and len(edges):
                    vis = edges[(edges >= a_lo) & (edges < a_hi)]
                    for ei in vis:
                        ax.axvline(
                            ei * MS_PER_SLICE, color="red",
                            linewidth=0.6, alpha=0.55, zorder=1,
                        )

            if show_beam:
                ax_b = ax.twinx()
                _beam_twins[ic_idx] = ax_b
                for si, (sid, data) in enumerate(session_data.items()):
                    beam = data.get("beam")
                    if beam is None:
                        continue
                    n = len(beam)
                    a_lo = min(lo, n)
                    a_hi = min(hi, n)
                    if a_hi <= a_lo:
                        continue
                    b_win = beam[a_lo:a_hi]
                    w_len = a_hi - a_lo
                    if w_len > DETAIL_MAX_POINTS:
                        step = max(1, w_len // DETAIL_MAX_POINTS)
                        bx = np.arange(a_lo, a_hi, step) * MS_PER_SLICE
                        by = b_win[::step]
                    else:
                        bx = np.arange(a_lo, a_hi) * MS_PER_SLICE
                        by = b_win
                    beam_label = f"Beam I ({sid})" if multi else "Beam I"
                    beam_color = sess_colors[si] if multi else ic_detail_colors[ic_idx]
                    ax_b.plot(
                        bx, by,
                        color=beam_color, linewidth=0.4, linestyle=(0, (4, 3)),
                        label=beam_label if ic_idx == 0 else None,
                    )
                ax_b.set_ylabel("Beam I", fontsize=8)
                ax_b.tick_params(axis="y", labelsize=7)

                ic_lo, ic_hi = ax.get_ylim()
                b_lo, b_hi = ax_b.get_ylim()
                if ic_hi != ic_lo and b_hi != b_lo:
                    ic_frac = -ic_lo / (ic_hi - ic_lo)
                    b_frac = -b_lo / (b_hi - b_lo)
                    if abs(ic_frac - b_frac) > 1e-6:
                        if ic_frac > b_frac:
                            new_b_lo = -ic_frac * (b_hi - b_lo) / (1 - ic_frac) if ic_frac < 1 else b_lo
                            ax_b.set_ylim(new_b_lo, b_hi)
                        else:
                            new_b_hi = -b_lo * (1 - ic_frac) / ic_frac if ic_frac > 0 else b_hi
                            ax_b.set_ylim(b_lo, new_b_hi)

            for sid_data in session_data.values():
                for offset, energy in sid_data["layer_boundaries"]:
                    offset_t = offset * MS_PER_SLICE
                    if lo_t < offset_t < hi_t:
                        ax.axvline(offset_t, color="gray", linewidth=0.5, alpha=0.35)
                        if ic_idx == 0:
                            ax.text(
                                offset_t, 1.0, f" {energy:g} MeV",
                                transform=ax.get_xaxis_transform(),
                                fontsize=7, va="top", ha="left",
                                color="gray", alpha=0.7,
                            )

            ax.set_xlim(lo_t, hi_t)
            ax.set_ylabel(f"{label}", fontsize=9)
            ax.grid(**GRID_KW, which="major")
            ax.grid(which="minor", color="#e0e0e0", linewidth=0.3)

        _DIG_COLORS = ["#e67e22", "#3498db", "#2ecc71", "#9b59b6", "#e74c3c"]
        if ax_digital is not None:
            ax_digital.clear()

            all_dig_keys: list[str] = []
            for data in session_data.values():
                for k in data.get("digital", {}):
                    if k not in all_dig_keys:
                        all_dig_keys.append(k)

            lane_height = 1.15
            lane_gap = 0.25
            dig_handles: list = []
            dig_labels_leg: list[str] = []
            for lane_i, col_key in enumerate(all_dig_keys):
                y_base = lane_i * (lane_height + lane_gap)
                color = _DIG_COLORS[lane_i % len(_DIG_COLORS)]
                for si, (sid, data) in enumerate(session_data.items()):
                    entry = data.get("digital", {}).get(col_key)
                    if entry is None:
                        continue
                    arr, label = entry
                    n = len(arr)
                    a_lo = min(lo, n)
                    a_hi = min(hi, n)
                    if a_hi <= a_lo:
                        continue
                    win = arr[a_lo:a_hi]
                    w_len = a_hi - a_lo
                    if w_len > DETAIL_MAX_POINTS:
                        step = max(1, w_len // DETAIL_MAX_POINTS)
                        px = np.arange(a_lo, a_hi, step) * MS_PER_SLICE
                        py = win[::step]
                    else:
                        px = np.arange(a_lo, a_hi) * MS_PER_SLICE
                        py = win
                    sig_max = np.nanmax(np.abs(py)) if len(py) else 1.0
                    if sig_max > 0:
                        py = py / sig_max
                    ax_digital.fill_between(
                        px, y_base, y_base + py * lane_height,
                        step="post", alpha=0.55, color=color,
                        linewidth=0,
                    )
                    if lane_i < len(dig_labels_leg):
                        continue
                    sig_label = f"{label} ({sid})" if multi else label
                    dig_labels_leg.append(sig_label)
                    dig_handles.append(
                        plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.55)
                    )

            n_lanes = len(all_dig_keys) or 1
            ax_digital.set_xlim(lo_t, hi_t)
            ax_digital.set_ylim(-0.1, n_lanes * (lane_height + lane_gap))
            ax_digital.set_yticks([])
            ax_digital.set_ylabel("Digital", fontsize=8)
            ax_digital.grid(**GRID_KW, which="major")
            if dig_handles:
                ax_digital.legend(
                    dig_handles, dig_labels_leg,
                    loc="upper right", fontsize=7, ncol=len(dig_labels_leg),
                )
            plt.setp(ax_digital.get_xticklabels(), visible=False)

        for a in ax_detail:
            plt.setp(a.get_xticklabels(), visible=False)
        last_detail = ax_digital if ax_digital is not None else ax_detail[-1]
        _time_axis(last_detail, lo_t, hi_t)
        plt.setp(last_detail.get_xticklabels(), visible=True)
        for a in ax_detail:
            a.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        if ax_digital is not None:
            ax_digital.xaxis.set_minor_locator(mticker.AutoMinorLocator())

        handles, labels = ax_detail[0].get_legend_handles_labels()
        if show_beam and _beam_twins[0] is not None:
            bh, bl = _beam_twins[0].get_legend_handles_labels()
            handles += bh
            labels += bl
        if handles:
            ax_detail[0].legend(handles, labels, loc="upper right", fontsize=8)

        if show_pos:
            _pos_keys = {"ic1": ("ic1_x", "ic1_y"), "ic2": ("ic2_x", "ic2_y")}

            for ic_idx, ic in enumerate(ic_keys):
                ax_sc = ax_scatter[ic_idx]
                if ax_sc is None:
                    continue
                ax_sc.clear()

                xk, yk = _pos_keys.get(ic, (None, None))
                has_data = xk is not None

                if has_data:
                    for si, (sid, data) in enumerate(session_data.items()):
                        if not data.get("has_positions"):
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
                else:
                    ax_sc.text(
                        0.5, 0.5, "No position data",
                        transform=ax_sc.transAxes, ha="center", va="center",
                        fontsize=9, color="gray", alpha=0.6,
                    )

                ax_sc.axhline(y=0, **REFLINE_KW)
                ax_sc.axvline(x=0, **REFLINE_KW)
                ax_sc.grid(**GRID_KW)
                ax_sc.set_aspect("equal", adjustable="datalim")
                ax_sc.tick_params(labelsize=7)
                ax_sc.set_title(f"{ic_labels[ic_idx]} Position", fontsize=9)
                if ic_idx == n_detail - 1:
                    ax_sc.set_xlabel("X (mm)", fontsize=8)

        fig.canvas.draw_idle()

    _draw_detail(0, initial_end_t)

    def _on_select(xmin: float, xmax: float) -> None:
        _draw_detail(xmin, xmax)

    span = SpanSelector(
        ax_timeline, _on_select, "horizontal",
        useblit=True, interactive=True,
        props=dict(alpha=0.25, facecolor="gold"),
    )

    fig._scan_kit_span = span  # type: ignore[attr-defined]

    _ZOOM_FACTOR = 0.8

    def _on_scroll(event):
        if event.inaxes is not ax_timeline:
            return
        cur_lo, cur_hi = ax_timeline.get_xlim()
        rng = cur_hi - cur_lo
        cx = event.xdata if event.xdata is not None else (cur_lo + cur_hi) / 2
        if event.button == "up":
            new_rng = rng * _ZOOM_FACTOR
        elif event.button == "down":
            new_rng = rng / _ZOOM_FACTOR
        else:
            return
        new_rng = max(new_rng, 50 * MS_PER_SLICE)
        new_rng = min(new_rng, max_t)
        frac = (cx - cur_lo) / rng if rng > 0 else 0.5
        new_lo = cx - frac * new_rng
        new_hi = cx + (1 - frac) * new_rng
        new_lo = max(0, new_lo)
        new_hi = min(max_t, new_hi)
        ax_timeline.set_xlim(new_lo, new_hi)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", _on_scroll)

    plt.show()
