"""IC timeslice replay: media-player style interactive current viewer."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.widgets import SpanSelector

from ..common import (
    C_ENERGY,
    C_IC1_CURRENT,
    C_IC2_CURRENT,
    C_IC3_CURRENT_A,
    C_IC3_CURRENT_B,
    C_IC3_CURRENT_C,
    C_IC3_CURRENT_D,
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

TIMELINE_BINS = 4000
DETAIL_MAX_POINTS = 80_000


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _resolve_col(columns, concept: str) -> str | None:
    """Resolve a concept to the actual column name present in *columns*."""
    return resolve_concept_column(columns, concept)


def _load_session_timeline(session_id: str, base_dir: str) -> dict | None:
    """Load and concatenate all timeslice frames into a unified timeline."""
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
    ts_ic1 = _resolve_col(df0.columns, C_IC1_CURRENT)
    ts_ic2 = _resolve_col(df0.columns, C_IC2_CURRENT)
    if not all([ts_layer, ts_ic1, ts_ic2]):
        return None

    # IC3 is optional (absent on G2 data)
    ts_ic3a = _resolve_col(df0.columns, C_IC3_CURRENT_A)
    ts_ic3b = _resolve_col(df0.columns, C_IC3_CURRENT_B)
    ts_ic3c = _resolve_col(df0.columns, C_IC3_CURRENT_C)
    ts_ic3d = _resolve_col(df0.columns, C_IC3_CURRENT_D)
    has_ic3 = all([ts_ic3a, ts_ic3b, ts_ic3c, ts_ic3d])

    # Position columns — try G3 key first, then G2
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

    ic1_parts: list[np.ndarray] = []
    ic2_parts: list[np.ndarray] = []
    ic3_parts: list[np.ndarray] = []
    pos_parts: dict[str, list[np.ndarray]] = {k: [] for k in ("ic1_x", "ic1_y", "ic2_x", "ic2_y")}
    energy_parts: list[np.ndarray] = []
    layer_boundaries: list[tuple[int, float]] = []
    offset = 0

    for df in frames:
        n = len(df)
        layer_id = df[ts_layer].iloc[0]
        energy = energy_by_layer.get(layer_id, 0.0)

        ic1_parts.append(df[ts_ic1].values)
        ic2_parts.append(df[ts_ic2].values)
        if has_ic3:
            ic3_parts.append(
                df[ts_ic3a].values
                + df[ts_ic3b].values
                + df[ts_ic3c].values
                + df[ts_ic3d].values
            )
        if has_positions:
            for label, col in pos_cols.items():
                pos_parts[label].append(df[col].values.astype(float))
        energy_parts.append(np.full(n, energy))
        layer_boundaries.append((offset, energy))
        offset += n

    if offset == 0:
        return None

    result: dict = {
        "ic1": np.concatenate(ic1_parts),
        "ic2": np.concatenate(ic2_parts),
        "layer_boundaries": layer_boundaries,
        "n_samples": offset,
        "has_ic3": has_ic3,
        "has_positions": has_positions,
        "energy": np.concatenate(energy_parts),
    }
    if has_ic3:
        result["ic3"] = np.concatenate(ic3_parts)
    if has_positions:
        result["ic1_x"] = transform.remap(np.concatenate(pos_parts["ic1_x"]), *transform.IC1_X_MAP)
        result["ic1_y"] = transform.remap(np.concatenate(pos_parts["ic1_y"]), *transform.IC1_Y_MAP)
        result["ic2_x"] = transform.remap(np.concatenate(pos_parts["ic2_x"]), *transform.IC2_X_MAP)
        result["ic2_y"] = transform.remap(np.concatenate(pos_parts["ic2_y"]), *transform.IC2_Y_MAP)
        # Raw values -1 or 0 indicate invalid fits; after remap they fall
        # outside the valid strip range [-128, 128].  NaN them out so
        # downstream scatter plots skip them automatically.
        _POS_LIMIT = transform.IC_MM_MAX  # 128
        for k in ("ic1_x", "ic1_y", "ic2_x", "ic2_y"):
            arr = result[k]
            arr[np.abs(arr) > _POS_LIMIT] = np.nan
    return result


def _compress_minmax(signal: np.ndarray, n_bins: int):
    """Compress signal into a min/max envelope for fast timeline rendering.

    Returns ``(x_centers, y_min, y_max)`` arrays of length *n_bins*.
    Falls back to the raw signal when it already fits.
    Uses vectorized ``reduceat`` operations for speed on large arrays.
    """
    n = len(signal)
    if n <= n_bins * 2:
        x = np.arange(n, dtype=float)
        return x, signal.copy(), signal.copy()

    bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
    starts = bin_edges[:-1]
    x = (starts + bin_edges[1:]) * 0.5

    safe = np.where(np.isnan(signal), np.inf, signal)
    y_min = np.minimum.reduceat(safe, starts)

    safe_max = np.where(np.isnan(signal), -np.inf, signal)
    y_max = np.maximum.reduceat(safe_max, starts)

    return x, y_min, y_max


# ---------------------------------------------------------------------------
# View entry point
# ---------------------------------------------------------------------------

def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Launch the IC timeslice replay viewer."""
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
    show_pos = any(d.get("has_positions", False) for d in session_data.values())

    ic_keys: list[str] = ["ic1", "ic2"]
    ic_labels: list[str] = ["IC1", "IC2"]
    ic_detail_colors: list[str] = ["#1f77b4", "#d62728"]
    if show_ic3:
        ic_keys.append("ic3")
        ic_labels.append("IC3 (A+B+C+D)")
        ic_detail_colors.append("#2ca02c")

    n_detail = len(ic_keys)

    # -- Layout ----------------------------------------------------------------
    # Each IC row: [line chart | scatter].  Bottom row: timeline spans full width.
    n_cols = 2 if show_pos else 1
    width_ratios = [4, 1] if show_pos else [1]

    fig = plt.figure(figsize=(22 if show_pos else 18, 10))
    fig.suptitle("IC Timeslice Replay", **SUPTITLE_KW)

    gs = gridspec.GridSpec(
        n_detail + 1, n_cols,
        height_ratios=[1] * n_detail + [0.30],
        width_ratios=width_ratios,
        hspace=0.18, wspace=0.04,
        top=0.94, bottom=0.06, left=0.03, right=0.99,
    )

    ax_detail = [fig.add_subplot(gs[0, 0])]
    for i in range(1, n_detail):
        ax_detail.append(fig.add_subplot(gs[i, 0], sharex=ax_detail[0]))
    ax_timeline = fig.add_subplot(gs[n_detail, :])

    ax_scatter: list[plt.Axes | None] = [None] * n_detail
    if show_pos:
        for i in range(n_detail):
            ax_scatter[i] = fig.add_subplot(gs[i, 1])

    for ax in ax_detail[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    # -- Compressed timeline ---------------------------------------------------
    for si, (sid, data) in enumerate(session_data.items()):
        x, y_min, y_max = _compress_minmax(data["ic1"], TIMELINE_BINS)
        label = f"IC1 — {sid}" if multi else "IC1"
        ax_timeline.fill_between(
            x, y_min, y_max, alpha=0.45,
            color=sess_colors[si], label=label,
        )

    ax_timeline.set_xlim(0, max_n)
    ax_timeline.set_ylabel("IC1", fontsize=8)
    ax_timeline.set_xlabel("Sample index")
    ax_timeline.grid(**GRID_KW)
    ax_timeline.tick_params(labelsize=8)
    if multi:
        ax_timeline.legend(loc="upper right", fontsize=7, ncol=len(loaded_ids))

    for sid_data in session_data.values():
        for offset, _energy in sid_data["layer_boundaries"]:
            if offset > 0:
                ax_timeline.axvline(offset, color="gray", linewidth=0.3, alpha=0.25)

    # -- Energy colormap for scatter -------------------------------------------
    all_energies = set()
    for data in session_data.values():
        all_energies.update(np.unique(data["energy"]))
    if all_energies:
        e_min, e_max = min(all_energies), max(all_energies)
    else:
        e_min, e_max = 0, 1
    norm = mcolors.Normalize(vmin=e_min, vmax=e_max)

    # -- Detail + scatter redraw callback --------------------------------------
    initial_end = min(max_n, max(2000, max_n // 10))

    def _draw_detail(xmin: float, xmax: float) -> None:
        lo = max(0, int(xmin))
        hi = min(max_n, int(xmax))
        if hi <= lo:
            return

        for ic_idx, (ic, label) in enumerate(zip(ic_keys, ic_labels)):
            ax = ax_detail[ic_idx]
            ax.clear()

            for si, (sid, data) in enumerate(session_data.items()):
                sig = data.get(ic)
                if sig is None:
                    continue
                n = len(sig)
                a_lo = min(lo, n)
                a_hi = min(hi, n)
                if a_hi <= a_lo:
                    continue

                window = sig[a_lo:a_hi]
                w_len = a_hi - a_lo

                if w_len > DETAIL_MAX_POINTS:
                    step = max(1, w_len // DETAIL_MAX_POINTS)
                    plot_x = np.arange(a_lo, a_hi, step)
                    plot_y = window[::step]
                else:
                    plot_x = np.arange(a_lo, a_hi)
                    plot_y = window

                line_color = sess_colors[si] if multi else ic_detail_colors[ic_idx]
                ax.plot(
                    plot_x, plot_y,
                    color=line_color, linewidth=0.5, alpha=0.85,
                    label=sid if (multi and ic_idx == 0) else None,
                )

            for sid_data in session_data.values():
                for offset, energy in sid_data["layer_boundaries"]:
                    if lo < offset < hi:
                        ax.axvline(offset, color="gray", linewidth=0.5, alpha=0.35)
                        if ic_idx == 0:
                            ax.text(
                                offset, 1.0, f" {energy:g} MeV",
                                transform=ax.get_xaxis_transform(),
                                fontsize=7, va="top", ha="left",
                                color="gray", alpha=0.7,
                            )

            ax.set_xlim(lo, hi)
            ax.set_ylabel(f"{label}", fontsize=9)
            ax.grid(**GRID_KW)

        for a in ax_detail[:-1]:
            plt.setp(a.get_xticklabels(), visible=False)
        ax_detail[-1].set_xlabel("Sample index")

        if multi:
            ax_detail[0].legend(loc="upper right", fontsize=8)

        # -- Per-row scatter plots for the same window --------------------------
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

    _draw_detail(0, initial_end)

    # -- Interactive brush on the timeline -------------------------------------
    def _on_select(xmin: float, xmax: float) -> None:
        _draw_detail(xmin, xmax)

    span = SpanSelector(
        ax_timeline, _on_select, "horizontal",
        useblit=True, interactive=True,
        props=dict(alpha=0.25, facecolor="gold"),
    )

    fig._scan_kit_span = span  # type: ignore[attr-defined]

    # -- Scroll-wheel zoom on the timeline X axis ------------------------------
    _ZOOM_FACTOR = 0.8  # each scroll tick zooms by 20%

    def _on_scroll(event):
        if event.inaxes is not ax_timeline:
            return
        cur_lo, cur_hi = ax_timeline.get_xlim()
        rng = cur_hi - cur_lo
        # Zoom centred on the cursor position
        cx = event.xdata if event.xdata is not None else (cur_lo + cur_hi) / 2
        if event.button == "up":
            new_rng = rng * _ZOOM_FACTOR
        elif event.button == "down":
            new_rng = rng / _ZOOM_FACTOR
        else:
            return
        new_rng = max(new_rng, 50)  # don't zoom past ~50 samples
        new_rng = min(new_rng, max_n)
        frac = (cx - cur_lo) / rng if rng > 0 else 0.5
        new_lo = cx - frac * new_rng
        new_hi = cx + (1 - frac) * new_rng
        new_lo = max(0, new_lo)
        new_hi = min(max_n, new_hi)
        ax_timeline.set_xlim(new_lo, new_hi)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", _on_scroll)

    plt.show()
