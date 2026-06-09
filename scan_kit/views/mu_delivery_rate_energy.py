"""Effective MU delivery rate vs beam energy (wall-clock time per layer)."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

from ..common import (
    C_CHARGE_REQ,
    C_ENERGY,
    C_TIME_NS,
    C_TIME_S,
    C_TIMESTAMP,
    DEFAULT_SESSION_COLORS,
    GRID_KW,
    SCATTER_SIZE,
    finish_view,
    resolve_concept_column,
    style_energy_axes,
    trend_session_prefix,
    view_grid,
)
from ..common.session_source import (
    load_session_csv,
    resolve_session_source,
)

_log = logging.getLogger(__name__)

VIEW_TITLE = "Effective MU Delivery Rate vs Energy"
Y_LABEL = "Delivery rate (MU/s)"
MIN_LAYER_DURATION_S = 0.05
_CURVE_SAMPLES = 200


def _spot_wall_time_s(spot_df: pd.DataFrame) -> np.ndarray | None:
    """Return per-spot wall time in seconds (absolute or session-relative)."""
    if "datetime" in spot_df.columns:
        dt = pd.to_datetime(spot_df["datetime"], errors="coerce", utc=True)
        if dt.notna().any():
            return dt.astype("int64").to_numpy(dtype=float) / 1e9

    col_s = resolve_concept_column(spot_df.columns, C_TIME_S)
    col_ns = resolve_concept_column(spot_df.columns, C_TIME_NS)
    if col_s is not None and col_ns is not None:
        sec = pd.to_numeric(spot_df[col_s], errors="coerce").to_numpy(dtype=float)
        nsec = pd.to_numeric(spot_df[col_ns], errors="coerce").to_numpy(dtype=float)
        return sec + nsec * 1e-9

    col_ts = resolve_concept_column(spot_df.columns, C_TIMESTAMP)
    if col_ts is not None:
        return pd.to_numeric(spot_df[col_ts], errors="coerce").to_numpy(dtype=float) / 1000.0

    return None


def _load_energy_rates(session_id: str, base_dir: str) -> dict[str, np.ndarray] | None:
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    input_map = load_session_csv(src, "input_map.csv")
    spot_data = load_session_csv(src, "spot_data.csv")
    if input_map is None or spot_data is None:
        return None

    col_charge = resolve_concept_column(input_map.columns, C_CHARGE_REQ)
    col_energy = resolve_concept_column(input_map.columns, C_ENERGY)
    if col_charge is None or col_energy is None:
        return None

    n = min(len(input_map), len(spot_data))
    wall_t = _spot_wall_time_s(spot_data.iloc[:n])
    if wall_t is None:
        return None

    charge = pd.to_numeric(input_map[col_charge].iloc[:n], errors="coerce").values
    ok_t = np.isfinite(wall_t)
    total_mu = float(np.nansum(charge[np.isfinite(charge)]))
    session_span_s = float(wall_t[ok_t].max() - wall_t[ok_t].min()) if ok_t.sum() >= 2 else 0.0
    if total_mu <= 0 or session_span_s < MIN_LAYER_DURATION_S:
        return None

    df = pd.DataFrame({
        "energy": pd.to_numeric(input_map[col_energy].iloc[:n], errors="coerce").values,
        "charge": charge,
        "wall_t": wall_t,
    })

    energies: list[float] = []
    rates: list[float] = []
    layer_mu: list[float] = []
    for energy, group in df.groupby("energy", sort=True):
        t = group["wall_t"].to_numpy(dtype=float)
        ok = np.isfinite(t)
        if ok.sum() < 2:
            continue
        duration_s = float(t[ok].max() - t[ok].min())
        if duration_s < MIN_LAYER_DURATION_S:
            continue

        mu = float(group["charge"].sum())
        if not np.isfinite(energy) or not np.isfinite(mu) or mu <= 0:
            continue

        energies.append(float(energy))
        layer_mu.append(mu)
        rates.append(mu / duration_s)

    if not energies:
        return None

    order = np.argsort(energies)
    return {
        "energy": np.asarray(energies, dtype=float)[order],
        "mu_rate": np.asarray(rates, dtype=float)[order],
        "layer_mu": np.asarray(layer_mu, dtype=float)[order],
        "session_avg_rate": total_mu / session_span_s,
    }


def _smooth_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotonic smooth curve through *(x, y)* using PCHIP."""
    if x.size < 2:
        return x, y
    x_fine = np.linspace(float(x[0]), float(x[-1]), _CURVE_SAMPLES)
    y_fine = PchipInterpolator(x, y)(x_fine)
    return x_fine, y_fine


def _make_avg_rate_legend(ax, entries: list[tuple[str, str]]) -> None:
    """Dashed-line legend for session-average delivery rates."""
    if not entries:
        return
    handles = [
        Line2D([0], [0], color=color, label=text, linewidth=1.4, linestyle="--")
        for text, color in entries
    ]
    legend = ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)
    ax.add_artist(legend)


def _plot_sessions(
    ax,
    session_data: dict[str, dict[str, np.ndarray]],
    energies: list[float],
    loaded_ids: list[str],
    colors: list[str],
) -> list[tuple[str, str]]:
    energy_index = {e: i for i, e in enumerate(energies)}
    avg_entries: list[tuple[str, str]] = []
    n_sessions = len(loaded_ids)

    for i, (sid, color) in enumerate(zip(loaded_ids, colors)):
        data = session_data[sid]
        xs = np.array([energy_index[e] for e in data["energy"]], dtype=float)
        ys = data["mu_rate"]

        if xs.size >= 2:
            x_curve, y_curve = _smooth_curve(xs, ys)
            ax.plot(
                x_curve,
                y_curve,
                color=color,
                linewidth=2.0,
                solid_capstyle="round",
                zorder=2,
            )
        ax.scatter(
            xs,
            ys,
            color=color,
            s=SCATTER_SIZE,
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )

        avg = float(data["session_avg_rate"])
        ax.axhline(avg, color=color, linestyle="--", linewidth=1.4, alpha=0.9, zorder=1)
        prefix = trend_session_prefix(sid, n_sessions=n_sessions)
        avg_entries.append((f"{prefix}{avg:.3g} MU/s avg", color))

    return avg_entries


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    del settings

    if not session_ids:
        return

    session_data = {
        sid: rates
        for sid in session_ids
        if (rates := _load_energy_rates(sid, base_dir)) is not None
    }
    if not session_data:
        _log.debug("No layer delivery-rate data for selected sessions")
        return

    energies = sorted({
        float(e)
        for data in session_data.values()
        for e in data["energy"]
    })
    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    # Single wide energy axis (many energy categories on x).
    fig, ax = view_grid(1, 1, squeeze=True, cell_w=13.0, cell_h=6.5)
    avg_entries = _plot_sessions(ax, session_data, energies, loaded_ids, colors)

    ax.set_title("Wall-clock delivery rate by energy")
    style_energy_axes(ax, energies, ylabel=Y_LABEL)
    ax.grid(**GRID_KW)
    _make_avg_rate_legend(ax, avg_entries)

    finish_view(fig, VIEW_TITLE, loaded_ids, colors, base_dir=base_dir)
    plt.show()
