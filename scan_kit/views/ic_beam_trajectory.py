"""Per-spot IC beam lines in X and Y — raw IC positions, extended along the beam.

Beam path (+z is **downstream**, direction of travel):

    scan magnet  →  ~1–3 m air  →  IC2  →  IC_SEP  →  IC1
      (upstream)                     z = 0              z = IC1_Z

Uses **raw** IC positions (registers remapped to mm at each chamber plane).
Non-raw (processed) positions are already forward-projected to isocenter, so
IC1 and IC2 values sit in the same reference plane and hide the beam angle
between chambers — only raw per-IC mm preserves the trajectory.

Each spot draws a line through (IC2, ic2) and (IC1, ic1), extended upstream
toward the scan magnets and downstream past IC1.  X and Y are plotted separately.
G2 raw data keeps register 64.5 → 0 mm on IC1; IC2 strip direction (0→127 vs
127→0) is corrected in :func:`process_position_data` without flipping IC1.

**IC alignment correction.** After remapping to mm, each chamber still carries a
constant offset relative to the true beam axis (mechanical/electrical alignment).
For each session and axis we subtract a rigid per-IC offset — the median of that
chamber's raw-mm cloud — so the per-spot lines form a frustum whose pivot sits on
the beam axis (position ≈ 0).  This leaves the per-spot angle and the pivot
*distance* unchanged (both depend only on slope differences) and only removes the
fake common tilt from misalignment.  The subtracted offsets are the measured IC
alignment errors and are reported per session in the panel legend.  Applies to
both G2 and G3 raw positions.

**Plan overlay.** Nominal ``x_position`` / ``y_position`` from the input map
(iso-center mm) are projected along rays from the measured scan-magnet pivot
through a **fitted iso plane** downstream of IC2 (median per-spot *z* where each
alignment-corrected measured line crosses its plan nominal; MAD uncertainty).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from ..common import (
    C_X_POSITION,
    C_Y_POSITION,
    process_position_data,
    try_load_position_data,
    set_view_header,
    format_session_legend_label,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_1x2,
    apply_tight_layout,
    GRID_KW,
    REFLINE_KW,
)
from ..common.session_notes import load_notes

_log = logging.getLogger(__name__)

# Plot origin at IC2 (first chamber along the downstream beam).
IC2_Z_MM = 0.0
IC1_Z_MM = 100.0
IC_SEP_MM = IC1_Z_MM - IC2_Z_MM

# Show magnet/air region upstream of IC2 and beam path downstream of IC1.
EXTEND_UPSTREAM_MM = 2000.0
EXTEND_DOWNSTREAM_MM = 2000.0

# Upstream view is clipped to this margin past the furthest pivot (mm).
PIVOT_MARGIN_MM = 100.0

Z_AXIS_LABEL = "Distance downstream from IC2 (mm)"

LINE_ALPHA = 0.1
LINE_LW = 0.5

PLAN_LINE_ALPHA = 0.35
PLAN_LINE_LW = 0.45

IC_PLANE_KW = dict(color="0.55", linestyle=":", linewidth=1.2, alpha=0.85, zorder=2)
MAGNET_LINE_KW = dict(linestyle="--", linewidth=1.4, alpha=0.95, zorder=4)
ISO_LINE_KW = dict(linestyle="-.", linewidth=1.1, alpha=0.85, zorder=3)
PLAN_LINE_KW = dict(linestyle=":", linewidth=PLAN_LINE_LW, alpha=PLAN_LINE_ALPHA, zorder=0)

# Cap spots used for pairwise crossover (n^2 pairs); fixed seed for reproducibility.
_PAIRWISE_MAX_SPOTS = 400
_PAIRWISE_MIN_INTERSECTIONS = 20
_PAIRWISE_RNG_SEED = 0
_ISO_MIN_ESTIMATES = 20
_SLOPE_EPS = 1e-9


@dataclass(frozen=True)
class _MagnetFit:
    """Upstream crossover of per-spot IC2–IC1 lines for one session axis."""

    z_pivot: float
    upstream_mm: float
    upstream_sigma_mm: float

    @property
    def is_valid(self) -> bool:
        return (
            np.isfinite(self.z_pivot)
            and np.isfinite(self.upstream_mm)
            and self.z_pivot < IC2_Z_MM
        )


@dataclass(frozen=True)
class _IsoFit:
    """Downstream iso plane where measured trajectories match plan nominals."""

    z_iso: float
    downstream_mm: float
    downstream_sigma_mm: float

    @property
    def is_valid(self) -> bool:
        return np.isfinite(self.z_iso) and self.z_iso > IC2_Z_MM


def _pairwise_upstream_crossings(p2: np.ndarray, slopes: np.ndarray) -> np.ndarray:
    """Z downstream from IC2 where pairs of spot lines intersect, upstream only."""
    n = p2.size
    z_vals: list[np.ndarray] = []
    for i in range(n - 1):
        dm = slopes[i] - slopes[i + 1 :]
        valid = np.abs(dm) > _SLOPE_EPS
        if not np.any(valid):
            continue
        z_ij = IC2_Z_MM + (p2[i + 1 :] - p2[i])[valid] / dm[valid]
        z_vals.append(z_ij[z_ij < IC2_Z_MM])
    if not z_vals:
        return np.array([], dtype=float)
    return np.concatenate(z_vals)


def _fit_magnet_pivot(p2: np.ndarray, p1: np.ndarray) -> _MagnetFit:
    """Median upstream crossover z from pairwise intersections of spot lines."""
    nan = _MagnetFit(float("nan"), float("nan"), float("nan"))
    if p2.size < 2:
        return nan

    slopes = (p1 - p2) / IC_SEP_MM
    n = p2.size
    if n > _PAIRWISE_MAX_SPOTS:
        rng = np.random.default_rng(_PAIRWISE_RNG_SEED)
        pick = np.sort(rng.choice(n, size=_PAIRWISE_MAX_SPOTS, replace=False))
        p2 = p2[pick]
        slopes = slopes[pick]

    z_cross = _pairwise_upstream_crossings(p2, slopes)
    if z_cross.size < _PAIRWISE_MIN_INTERSECTIONS:
        return nan

    z_pivot = float(np.median(z_cross))
    mad = float(np.median(np.abs(z_cross - z_pivot)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(z_cross, ddof=1))

    return _MagnetFit(
        z_pivot,
        IC2_Z_MM - z_pivot,
        float(sigma) if np.isfinite(sigma) else float("nan"),
    )


def _per_spot_iso_estimates(
    p2: np.ndarray,
    p1: np.ndarray,
    plan_p: np.ndarray,
) -> np.ndarray:
    """Z downstream from IC2 where each measured line crosses its plan nominal."""
    slopes = (p1 - p2) / IC_SEP_MM
    valid = (
        np.isfinite(p2)
        & np.isfinite(plan_p)
        & np.isfinite(slopes)
        & (np.abs(slopes) > _SLOPE_EPS)
    )
    z_iso = IC2_Z_MM + (plan_p[valid] - p2[valid]) / slopes[valid]
    return z_iso[np.isfinite(z_iso) & (z_iso > IC2_Z_MM)]


def _fit_iso_plane(
    p2: np.ndarray,
    p1: np.ndarray,
    plan_p: np.ndarray,
    z_pivot: float,
) -> _IsoFit:
    """Median downstream *z* where each measured line crosses its plan nominal.

    Analogous to the upstream pivot fit: each spot contributes one crossing
    estimate; the median is robust to outliers.  (A global IC2/IC1 residual
    search is flat in *z_iso* because plan rays share one scale factor.)
    """
    nan = _IsoFit(float("nan"), float("nan"), float("nan"))
    if p2.size < 2 or not np.isfinite(z_pivot):
        return nan

    ok = np.isfinite(p2) & np.isfinite(p1) & np.isfinite(plan_p)
    p2 = p2[ok]
    p1 = p1[ok]
    plan_p = plan_p[ok]
    if p2.size < 2:
        return nan

    z_spot = _per_spot_iso_estimates(p2, p1, plan_p)
    if z_spot.size < _ISO_MIN_ESTIMATES:
        return nan

    n = z_spot.size
    if n > _PAIRWISE_MAX_SPOTS:
        rng = np.random.default_rng(_PAIRWISE_RNG_SEED)
        z_spot = np.sort(rng.choice(z_spot, size=_PAIRWISE_MAX_SPOTS, replace=False))

    z_iso = float(np.median(z_spot))
    mad = float(np.median(np.abs(z_spot - z_iso)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(z_spot, ddof=1))

    return _IsoFit(
        z_iso,
        z_iso - IC2_Z_MM,
        float(sigma) if np.isfinite(sigma) else float("nan"),
    )


def _ic_alignment_offsets(p2: np.ndarray, p1: np.ndarray) -> tuple[float, float]:
    """Rigid per-IC offset (median raw mm) = chamber alignment vs the beam axis.

    Subtracting these centres each chamber's cloud so the per-spot frustum pivots
    on-axis.  Slope differences (hence the per-spot angle and the pivot distance)
    are unchanged; only the fake common tilt from misalignment is removed.
    """
    return float(np.median(p2)), float(np.median(p1))


def _format_magnet_legend_label(
    session_id: str,
    fit: _MagnetFit,
    iso: _IsoFit | None,
    off2: float,
    off1: float,
    *,
    notes: dict[str, str] | None,
) -> str:
    sid = format_session_legend_label(session_id, notes)
    dist = fit.upstream_mm
    sig = fit.upstream_sigma_mm
    if not np.isfinite(dist):
        head = f"{sid}: n/a"
    elif np.isfinite(sig) and sig > 0:
        head = f"{sid}: pivot {dist:.0f} \u00b1 {sig:.0f} mm up"
    else:
        head = f"{sid}: pivot {dist:.0f} mm up"
    lines = [head, f"   align: IC1 {off1:+.1f}, IC2 {off2:+.1f} mm"]
    if iso is not None and iso.is_valid:
        iso_sig = iso.downstream_sigma_mm
        if np.isfinite(iso_sig) and iso_sig > 0:
            lines.append(
                f"   iso {iso.downstream_mm:.0f} \u00b1 {iso_sig:.0f} mm down",
            )
        else:
            lines.append(f"   iso {iso.downstream_mm:.0f} mm down")
    return "\n".join(lines)


def _process_session(session_id: str, position_key: str, base_dir: str):
    return process_position_data(
        session_id,
        position_key,
        base_dir=base_dir,
        extra_input_columns=[C_X_POSITION, C_Y_POSITION],
    )


def _valid_spot_mask(data: dict) -> np.ndarray:
    x2 = np.asarray(data["ic2_x"], dtype=float)
    y2 = np.asarray(data["ic2_y"], dtype=float)
    x1 = np.asarray(data["ic1_x"], dtype=float)
    y1 = np.asarray(data["ic1_y"], dtype=float)
    return np.isfinite(x2) & np.isfinite(y2) & np.isfinite(x1) & np.isfinite(y1)


def _spot_segments(p2: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Line through IC2/IC1 samples, extended upstream and downstream along +z."""
    slope = (p1 - p2) / IC_SEP_MM
    z_upstream = IC2_Z_MM - EXTEND_UPSTREAM_MM
    z_downstream = IC1_Z_MM + EXTEND_DOWNSTREAM_MM
    p_upstream = p2 + slope * (z_upstream - IC2_Z_MM)
    p_downstream = p1 + slope * (z_downstream - IC1_Z_MM)
    return np.stack(
        [
            np.column_stack([np.full(p2.shape, z_upstream), p_upstream]),
            np.column_stack([np.full(p2.shape, z_downstream), p_downstream]),
        ],
        axis=1,
    )


def _project_plan_to_z(
    plan_p: np.ndarray, z_pivot: float, z_iso: float, z: float,
) -> np.ndarray:
    """Project iso-center plan position to *z* via ray from pivot through iso."""
    return plan_p * (z - z_pivot) / (z_iso - z_pivot)


def _plan_segments(plan_p: np.ndarray, z_pivot: float, z_iso: float) -> np.ndarray:
    """Per-spot plan rays from iso through the pivot, extended along +z."""
    z_upstream = IC2_Z_MM - EXTEND_UPSTREAM_MM
    z_downstream = IC1_Z_MM + EXTEND_DOWNSTREAM_MM
    return np.stack(
        [
            np.column_stack([
                np.full(plan_p.shape, z_upstream),
                _project_plan_to_z(plan_p, z_pivot, z_iso, z_upstream),
            ]),
            np.column_stack([
                np.full(plan_p.shape, z_downstream),
                _project_plan_to_z(plan_p, z_pivot, z_iso, z_downstream),
            ]),
        ],
        axis=1,
    )


def _draw_plan_lines(ax, plan_p: np.ndarray, z_pivot: float, z_iso: float, color) -> int:
    if plan_p.size == 0 or not np.isfinite(z_pivot) or not np.isfinite(z_iso):
        return 0
    if abs(z_iso - z_pivot) < 1e-3:
        return 0
    lc = LineCollection(
        _plan_segments(plan_p, z_pivot, z_iso),
        colors=color,
        **PLAN_LINE_KW,
        capstyle="round",
    )
    ax.add_collection(lc)
    return int(plan_p.size)


def _draw_spot_lines(ax, p2: np.ndarray, p1: np.ndarray, color) -> int:
    if p2.size == 0:
        return 0
    lc = LineCollection(
        _spot_segments(p2, p1),
        colors=color,
        alpha=LINE_ALPHA,
        linewidths=LINE_LW,
        capstyle="round",
        zorder=1,
    )
    ax.add_collection(lc)
    return int(p2.size)


def _draw_ic_planes(ax):
    for z, label in ((IC2_Z_MM, "IC2"), (IC1_Z_MM, "IC1")):
        ax.axvline(z, **IC_PLANE_KW)
        ax.text(
            z, 0.98, label,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="0.45",
        )
    ax.text(
        0.02, 0.98, "\u2190 upstream (scan magnets)",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=8, color="0.45",
    )
    ax.text(
        0.98, 0.98, "downstream \u2192",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=8, color="0.45",
    )


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Draw per-spot raw IC trajectories in X and Y."""
    if not session_ids:
        _log.debug("No sessions selected")
        return

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        data = try_load_position_data(
            sid, base_dir, _process_session, raw=True,
        )  # non-raw is iso-projected; raw keeps each IC in its own plane
        if data is not None and _valid_spot_mask(data).any():
            session_data[sid] = data

    if not session_data:
        _log.debug("No valid IC1/IC2 position data found for any session")
        return

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    notes = load_notes(base_dir)

    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)

    axis_defs = (
        (ax_x, "X", "ic2_x", "ic1_x", C_X_POSITION),
        (ax_y, "Y", "ic2_y", "ic1_y", C_Y_POSITION),
    )

    pivot_z_by_axis: dict[int, list[float]] = {0: [], 1: []}
    magnet_legends: dict[int, list[tuple[str, str]]] = {0: [], 1: []}
    plan_drawn: dict[int, bool] = {0: False, 1: False}

    for sid, color in zip(loaded_ids, colors):
        data = session_data[sid]
        keep = _valid_spot_mask(data)
        for ax_idx, (ax, axis_name, ic2_key, ic1_key, plan_key) in enumerate(axis_defs):
            p2 = np.asarray(data[ic2_key], dtype=float)[keep]
            p1 = np.asarray(data[ic1_key], dtype=float)[keep]
            if p2.size == 0:
                _log.debug("Session %s: no %s lines drawn", sid, axis_name)
                continue

            # Rigid per-IC alignment correction: centre each chamber so the
            # frustum pivots on-axis (keeps slope, hence angle and distance).
            off2, off1 = _ic_alignment_offsets(p2, p1)
            p2 -= off2
            p1 -= off1

            n = _draw_spot_lines(ax, p2, p1, color)
            if n == 0:
                continue

            fit = _fit_magnet_pivot(p2, p1)
            iso_fit: _IsoFit | None = None
            if fit.is_valid:
                pivot_z_by_axis[ax_idx].append(fit.z_pivot)
                ax.axvline(fit.z_pivot, color=color, **MAGNET_LINE_KW)

                if plan_key in data:
                    plan_p = np.asarray(data[plan_key], dtype=float)[keep]
                    plan_ok = np.isfinite(plan_p)
                    if plan_ok.any():
                        iso_fit = _fit_iso_plane(
                            p2[plan_ok], p1[plan_ok], plan_p[plan_ok], fit.z_pivot,
                        )
                        if iso_fit.is_valid:
                            ax.axvline(iso_fit.z_iso, color=color, **ISO_LINE_KW)
                            _draw_plan_lines(
                                ax, plan_p[plan_ok], fit.z_pivot, iso_fit.z_iso, color,
                            )
                            plan_drawn[ax_idx] = True

                magnet_legends[ax_idx].append(
                    (
                        _format_magnet_legend_label(
                            sid, fit, iso_fit, off2, off1, notes=notes,
                        ),
                        color,
                    ),
                )
                _log.debug(
                    "Session %s %s pivot: %.0f ± %.0f mm up (z=%.1f); "
                    "iso: %s mm down; align IC1=%.2f IC2=%.2f",
                    sid,
                    axis_name,
                    fit.upstream_mm,
                    fit.upstream_sigma_mm,
                    fit.z_pivot,
                    (
                        f"{iso_fit.downstream_mm:.0f} ± {iso_fit.downstream_sigma_mm:.0f}"
                        if iso_fit is not None and iso_fit.is_valid
                        else "n/a"
                    ),
                    off1,
                    off2,
                )
            else:
                _log.debug(
                    "Session %s %s: no upstream line crossover found", sid, axis_name,
                )

    z_hi = IC1_Z_MM + EXTEND_DOWNSTREAM_MM

    for ax_idx, (ax, axis_name, _, _, _) in enumerate(axis_defs):
        # Clip the upstream side to just past the furthest pivot in this panel.
        pivots = pivot_z_by_axis[ax_idx]
        if pivots:
            z_lo = min(pivots) - PIVOT_MARGIN_MM
        else:
            z_lo = IC2_Z_MM - EXTEND_UPSTREAM_MM

        ax.axhline(y=0, **REFLINE_KW)
        _draw_ic_planes(ax)
        ax.autoscale_view(scalex=False)
        ax.set_xlim(z_lo, z_hi)
        ax.set_xlabel(Z_AXIS_LABEL)
        ax.set_ylabel(f"{axis_name} position (mm, raw IC)")
        ax.set_title(f"{axis_name} — per-spot IC lines")
        ax.grid(**GRID_KW)

        entries = magnet_legends[ax_idx]
        if entries:
            handles = [
                Line2D([0], [0], color=c, **MAGNET_LINE_KW, label=label)
                for label, c in entries
            ]
            if plan_drawn[ax_idx]:
                handles.append(
                    Line2D(
                        [0], [0],
                        color="0.35",
                        **PLAN_LINE_KW,
                        label="Plan (iso \u2194 pivot)",
                    ),
                )
                handles.append(
                    Line2D(
                        [0], [0],
                        color="0.35",
                        **ISO_LINE_KW,
                        label="Fitted iso plane",
                    ),
                )
            ax.legend(
                handles=handles,
                title="Pivot, iso plane & plan",
                loc="lower left",
                fontsize=8,
                title_fontsize=9,
                framealpha=0.92,
            )

    set_view_header(
        fig,
        (
            f"IC Beam Trajectory (raw IC; "
            f"{EXTEND_UPSTREAM_MM:g} mm upstream of IC2 / "
            f"{EXTEND_DOWNSTREAM_MM:g} mm downstream of IC1)"
        ),
        loaded_ids,
        colors,
        base_dir=base_dir,
    )

    apply_tight_layout()
    plt.show()
