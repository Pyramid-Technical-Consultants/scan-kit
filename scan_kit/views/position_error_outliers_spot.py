"""Position Error Outliers (Spot) — flag spots whose deviation from target is a clear statistical outlier.

Per-spot IC1/IC2 X and Y position errors (measured non-raw position minus the
prescribed ``X_POSITION``/``Y_POSITION``) are screened **per axis** with the
robust Iglewicz-Hoaglin modified z-score (median/MAD). A spot is flagged when
any of its four error components exceeds ``|M| > MOD_Z_THRESHOLD``, i.e. it
deviates far more than the bulk of spots in that same session.

Outliers are shown as a spatial map of target positions (offenders highlighted
and sized by severity) alongside a ranked table of the worst spots.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from ..common import (
    C_LAYER_ID,
    C_SPOT_NO,
    C_X_POSITION,
    C_Y_POSITION,
    CELL_SQUARE,
    DEFAULT_SESSION_COLORS,
    GRID_KW,
    REFLINE_KW,
    finish_view,
    process_position_data,
    try_load_position_data,
    view_grid,
)

_log = logging.getLogger(__name__)

#: Iglewicz-Hoaglin cutoff for a "clear" outlier on the modified z-score.
MOD_Z_THRESHOLD = 3.5
#: Most rows to list in the ranked table (worst offenders first).
MAX_TABLE_ROWS = 25

# Error components screened independently (label -> error-array key).
_COMPONENTS = (
    ("IC1 X", "ic1_x_err"),
    ("IC1 Y", "ic1_y_err"),
    ("IC2 X", "ic2_x_err"),
    ("IC2 Y", "ic2_y_err"),
)


def _process_session(session_id: str, position_key: str, base_dir: str):
    """Load non-raw spot positions and compute per-axis IC1/IC2 error vs plan."""
    data = process_position_data(
        session_id,
        position_key,
        extra_input_columns=[C_X_POSITION, C_Y_POSITION, C_LAYER_ID, C_SPOT_NO],
        base_dir=base_dir,
    )
    if data is None:
        return None
    if C_X_POSITION not in data or C_Y_POSITION not in data:
        _log.debug("Session %s: input_map missing plan position columns; skipping", session_id)
        return None

    n = np.asarray(data[C_X_POSITION], dtype=float).size
    plan_x = np.asarray(data[C_X_POSITION], dtype=float)
    plan_y = np.asarray(data[C_Y_POSITION], dtype=float)

    out = {
        "session_id": session_id,
        "plan_x": plan_x,
        "plan_y": plan_y,
        "energy": np.asarray(data["energy"], dtype=float),
        "layer": _layer_index(data.get(C_LAYER_ID), n),
        "spot": _as_int_array(data.get(C_SPOT_NO), n),
        "ic1_x_err": np.asarray(data["ic1_x"], dtype=float) - plan_x,
        "ic1_y_err": np.asarray(data["ic1_y"], dtype=float) - plan_y,
        "ic2_x_err": np.asarray(data["ic2_x"], dtype=float) - plan_x,
        "ic2_y_err": np.asarray(data["ic2_y"], dtype=float) - plan_y,
    }
    return out


def _as_int_array(values, n: int) -> np.ndarray:
    """Best-effort integer column (falls back to row index when absent/unparseable)."""
    if values is None:
        return np.arange(n)
    arr = np.asarray(values, dtype=float)
    if arr.size != n:
        return np.arange(n)
    out = np.where(np.isfinite(arr), arr, -1).astype(int)
    return out


def _layer_index(values, n: int) -> np.ndarray:
    """Map the opaque per-layer id to a 0-based ordinal by first appearance.

    Matches the ``layer-N`` session folder naming and is far more readable than
    the raw 9-10 digit layer id stored in ``input_map.csv``.
    """
    raw = _as_int_array(values, n)
    seen: dict[int, int] = {}
    out = np.empty(n, dtype=int)
    for i, v in enumerate(raw):
        idx = seen.get(v)
        if idx is None:
            idx = len(seen)
            seen[v] = idx
        out[i] = idx
    return out


def _modified_z(values: np.ndarray) -> np.ndarray:
    """Iglewicz-Hoaglin modified z-score: 0.6745*(x - median)/MAD (robust to outliers)."""
    v = np.asarray(values, dtype=float)
    z = np.full(v.shape, np.nan)
    finite = np.isfinite(v)
    if not finite.any():
        return z
    med = np.median(v[finite])
    mad = np.median(np.abs(v[finite] - med))
    if mad > 0:
        z[finite] = 0.6745 * (v[finite] - med) / mad
        return z
    # Degenerate spread (MAD == 0): fall back to mean-absolute-deviation scaling.
    mean = np.mean(v[finite])
    mean_ad = np.mean(np.abs(v[finite] - mean))
    if mean_ad > 0:
        z[finite] = (v[finite] - mean) / (1.253314 * mean_ad)
    else:
        z[finite] = 0.0
    return z


def _flag_outliers(session: dict) -> dict:
    """Add per-spot worst |modified z|, the triggering component, and an outlier mask."""
    n = session["plan_x"].size
    abs_z = np.zeros((len(_COMPONENTS), n))
    for i, (_label, key) in enumerate(_COMPONENTS):
        z = _modified_z(session[key])
        abs_z[i] = np.where(np.isfinite(z), np.abs(z), 0.0)

    worst_idx = np.argmax(abs_z, axis=0)
    worst_z = abs_z[worst_idx, np.arange(n)]
    session["worst_z"] = worst_z
    session["worst_component"] = worst_idx
    session["outlier_mask"] = worst_z > MOD_Z_THRESHOLD
    return session


def _outlier_records(session: dict) -> list[dict]:
    """Build table rows for every flagged spot in one session."""
    mask = session["outlier_mask"]
    records = []
    for idx in np.flatnonzero(mask):
        comp_i = int(session["worst_component"][idx])
        label, key = _COMPONENTS[comp_i]
        records.append(
            {
                "session_id": session["session_id"],
                "layer": int(session["layer"][idx]),
                "spot": int(session["spot"][idx]),
                "energy": float(session["energy"][idx]),
                "component": label,
                "error_mm": float(session[key][idx]),
                "z": float(session["worst_z"][idx]),
            }
        )
    return records


def _short_sid(session_id: str) -> str:
    return session_id if len(session_id) <= 8 else f"…{session_id[-6:]}"


def _draw_map(ax, sessions: list[dict], color_by_sid: dict) -> int:
    total = 0
    for session in sessions:
        ax.scatter(
            session["plan_x"], session["plan_y"],
            color="0.8", s=6, marker="o", edgecolors="none", zorder=1,
        )
    for session in sessions:
        mask = session["outlier_mask"]
        total += int(mask.sum())
        if not mask.any():
            continue
        color = color_by_sid[session["session_id"]]
        sizes = 28 + 18 * np.clip(session["worst_z"][mask] - MOD_Z_THRESHOLD, 0, 12)
        ax.scatter(
            session["plan_x"][mask], session["plan_y"][mask],
            s=sizes, facecolors="none", edgecolors=color, linewidths=1.4, zorder=3,
        )

    ax.axhline(0, **REFLINE_KW)
    ax.axvline(0, **REFLINE_KW)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Target X (mm)")
    ax.set_ylabel("Target Y (mm)")
    ax.set_title(f"Target positions — {total} outlier spot(s) highlighted")
    ax.grid(**GRID_KW)
    ax.margins(0.05)
    return total


def _draw_table(ax, records: list[dict], color_by_sid: dict) -> None:
    ax.axis("off")
    if not records:
        ax.text(0.5, 0.5, "No clear outliers found", ha="center", va="center",
                fontsize=11, color="0.4")
        return

    records = sorted(records, key=lambda r: r["z"], reverse=True)
    shown = records[:MAX_TABLE_ROWS]

    col_labels = ["Session", "Layer", "Spot", "E (MeV)", "Axis", "Err (mm)", "mod-z"]
    cell_text = [
        [
            _short_sid(r["session_id"]),
            str(r["layer"]),
            str(r["spot"]),
            f"{r['energy']:.1f}",
            r["component"],
            f"{r['error_mm']:+.2f}",
            f"{r['z']:.1f}",
        ]
        for r in shown
    ]
    row_colors = [color_by_sid[r["session_id"]] for r in shown]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.15)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.85")
        if row == 0:
            cell.set_facecolor("0.93")
            cell.set_text_props(fontweight="bold")
        elif col == 0:
            cell.get_text().set_color(row_colors[row - 1])

    n_total = len(records)
    suffix = f" (showing worst {MAX_TABLE_ROWS} of {n_total})" if n_total > MAX_TABLE_ROWS else ""
    ax.set_title(f"Worst offenders by modified z-score{suffix}", fontsize=10)


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Flag clear per-axis position outliers per session; show a map and ranked table."""
    if not session_ids:
        print("No sessions selected")
        return

    sessions: list[dict] = []
    for sid in session_ids:
        data = try_load_position_data(sid, base_dir, _process_session, raw=False)
        if data is not None:
            sessions.append(_flag_outliers(data))

    if not sessions:
        print("No valid spot position data found for any session")
        return

    loaded_ids = [s["session_id"] for s in sessions]
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    color_by_sid = dict(zip(loaded_ids, colors))

    records: list[dict] = []
    for session in sessions:
        records.extend(_outlier_records(session))

    fig, (ax_map, ax_table) = view_grid(
        1, 2, cell_w=CELL_SQUARE, cell_h=CELL_SQUARE, squeeze=True,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    _draw_map(ax_map, sessions, color_by_sid)
    _draw_table(ax_table, records, color_by_sid)

    finish_view(
        fig,
        f"Position Error Outliers (spot, |mod-z| > {MOD_Z_THRESHOLD})",
        loaded_ids,
        colors,
        base_dir=base_dir,
    )
    plt.show()
