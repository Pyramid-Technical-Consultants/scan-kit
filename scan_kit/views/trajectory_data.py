"""Load raw IC spot positions and build 3D trajectory geometry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..common import (
    C_X_POSITION,
    C_Y_POSITION,
    process_position_data,
    try_load_position_data,
)
from ..common.ic_trajectory import ic_alignment_offsets
from ..common.scan_magnet_model import (
    beam_lateral_mm,
    plan_lateral_at_z,
    segment_z_bounds,
)
from ..common.session_notes import load_notes
from ..common.trajectory_fits import (
    IsoFit,
    MagnetFit,
    combined_iso_z,
    combined_pivot_z,
    fit_iso_plane,
    fit_magnet_pivot,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrajectorySession:
    session_id: str
    ic2_x: np.ndarray
    ic2_y: np.ndarray
    ic1_x: np.ndarray
    ic1_y: np.ndarray
    plan_x: np.ndarray | None
    plan_y: np.ndarray | None
    energy: np.ndarray | None
    align_ic2_x: float
    align_ic1_x: float
    align_ic2_y: float
    align_ic1_y: float
    magnet_x: MagnetFit
    magnet_y: MagnetFit
    iso_x: IsoFit | None
    iso_y: IsoFit | None

    @property
    def n_spots(self) -> int:
        return int(self.ic2_x.size)

    @property
    def pivot_z(self) -> float:
        """Median X/Y dipole pivot depth (use per-axis fits for physics)."""
        return combined_pivot_z(self.magnet_x, self.magnet_y)

    @property
    def iso_z(self) -> float:
        """Median X/Y iso depth (use per-axis iso fits for physics)."""
        return combined_iso_z(self.iso_x, self.iso_y)


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


def build_trajectory_session(session_id: str, data: dict) -> TrajectorySession | None:
    keep = _valid_spot_mask(data)
    if not np.any(keep):
        return None

    x2 = np.asarray(data["ic2_x"], dtype=float)[keep]
    y2 = np.asarray(data["ic2_y"], dtype=float)[keep]
    x1 = np.asarray(data["ic1_x"], dtype=float)[keep]
    y1 = np.asarray(data["ic1_y"], dtype=float)[keep]

    off2_x, off1_x = ic_alignment_offsets(x2, x1)
    off2_y, off1_y = ic_alignment_offsets(y2, y1)
    x2_a = x2 - off2_x
    x1_a = x1 - off1_x
    y2_a = y2 - off2_y
    y1_a = y1 - off1_y

    magnet_x = fit_magnet_pivot(x2_a, x1_a)
    magnet_y = fit_magnet_pivot(y2_a, y1_a)

    plan_x = (
        np.asarray(data[C_X_POSITION], dtype=float)[keep]
        if C_X_POSITION in data
        else None
    )
    plan_y = (
        np.asarray(data[C_Y_POSITION], dtype=float)[keep]
        if C_Y_POSITION in data
        else None
    )
    energy = (
        np.asarray(data["energy"], dtype=float)[keep]
        if "energy" in data
        else None
    )

    iso_x: IsoFit | None = None
    iso_y: IsoFit | None = None
    if magnet_x.is_valid and plan_x is not None:
        plan_ok = np.isfinite(plan_x)
        if plan_ok.any():
            iso_x = fit_iso_plane(
                x2_a[plan_ok], x1_a[plan_ok], plan_x[plan_ok], magnet_x.z_pivot,
            )
    if magnet_y.is_valid and plan_y is not None:
        plan_ok = np.isfinite(plan_y)
        if plan_ok.any():
            iso_y = fit_iso_plane(
                y2_a[plan_ok], y1_a[plan_ok], plan_y[plan_ok], magnet_y.z_pivot,
            )

    return TrajectorySession(
        session_id=session_id,
        ic2_x=x2_a,
        ic2_y=y2_a,
        ic1_x=x1_a,
        ic1_y=y1_a,
        plan_x=plan_x,
        plan_y=plan_y,
        energy=energy,
        align_ic2_x=off2_x,
        align_ic1_x=off1_x,
        align_ic2_y=off2_y,
        align_ic1_y=off1_y,
        magnet_x=magnet_x,
        magnet_y=magnet_y,
        iso_x=iso_x,
        iso_y=iso_y,
    )


def load_trajectory_sessions(
    session_ids: Sequence[str],
    base_dir: str,
) -> dict[str, TrajectorySession]:
    out: dict[str, TrajectorySession] = {}
    for sid in session_ids:
        data = try_load_position_data(sid, base_dir, _process_session, raw=True)
        if data is None:
            continue
        session = build_trajectory_session(sid, data)
        if session is not None:
            out[sid] = session
    return out


def probe_trajectory_availability(session_ids: Sequence[str], base_dir: str) -> bool:
    return bool(load_trajectory_sessions(session_ids, base_dir))


def segment_z_extent(
    session: TrajectorySession,
    *,
    extend_upstream_mm: float,
    extend_downstream_mm: float,
) -> tuple[float, float]:
    """Clip range from furthest upstream dipole pivot to furthest iso plane."""
    return segment_z_bounds(
        session.magnet_x,
        session.magnet_y,
        session.iso_x,
        session.iso_y,
        extend_upstream_mm=extend_upstream_mm,
        extend_downstream_mm=extend_downstream_mm,
    )


def spot_segments_3d(
    session: TrajectorySession,
    *,
    extend_upstream_mm: float,
    extend_downstream_mm: float,
) -> np.ndarray:
    """Return ``(n_spots, 2, 3)`` line endpoints in mm (x, y, z downstream from IC2).

    When magnet and iso fits are available, rays run from the furthest upstream
    dipole pivot to the furthest iso plane.  Lateral position uses independent X/Y
    IC extrapolation (separate virtual pivots per scan dipole).
    """
    n = session.n_spots
    if n == 0:
        return np.empty((0, 2, 3), dtype=float)

    z_start, z_end = segment_z_extent(
        session,
        extend_upstream_mm=extend_upstream_mm,
        extend_downstream_mm=extend_downstream_mm,
    )

    x_start = beam_lateral_mm(session.ic2_x, session.ic1_x, z_start)
    y_start = beam_lateral_mm(session.ic2_y, session.ic1_y, z_start)
    x_end = beam_lateral_mm(session.ic2_x, session.ic1_x, z_end)
    y_end = beam_lateral_mm(session.ic2_y, session.ic1_y, z_end)

    segments = np.empty((n, 2, 3), dtype=float)
    segments[:, 0, 0] = x_start
    segments[:, 0, 1] = y_start
    segments[:, 0, 2] = z_start
    segments[:, 1, 0] = x_end
    segments[:, 1, 1] = y_end
    segments[:, 1, 2] = z_end
    return segments


def plan_segments_3d(
    session: TrajectorySession,
    *,
    extend_upstream_mm: float,
    extend_downstream_mm: float,
) -> np.ndarray | None:
    """Per-spot plan rays in 3D using separate X/Y dipole pivots and iso planes."""
    if session.plan_x is None or session.plan_y is None:
        return None

    z_px = session.magnet_x.z_pivot
    z_py = session.magnet_y.z_pivot
    z_ix = session.iso_x.z_iso if session.iso_x is not None else float("nan")
    z_iy = session.iso_y.z_iso if session.iso_y is not None else float("nan")

    if not (
        session.magnet_x.is_valid
        and session.magnet_y.is_valid
        and session.iso_x is not None
        and session.iso_x.is_valid
        and session.iso_y is not None
        and session.iso_y.is_valid
    ):
        return None

    plan_x = session.plan_x
    plan_y = session.plan_y
    ok = np.isfinite(plan_x) & np.isfinite(plan_y)
    if not np.any(ok):
        return None

    plan_x = plan_x[ok]
    plan_y = plan_y[ok]
    n = plan_x.size

    z_start, z_end = segment_z_extent(
        session,
        extend_upstream_mm=extend_upstream_mm,
        extend_downstream_mm=extend_downstream_mm,
    )

    segments = np.empty((n, 2, 3), dtype=float)
    segments[:, 0, 0] = plan_lateral_at_z(plan_x, z_px, z_ix, z_start)
    segments[:, 0, 1] = plan_lateral_at_z(plan_y, z_py, z_iy, z_start)
    segments[:, 0, 2] = z_start
    segments[:, 1, 0] = plan_x
    segments[:, 1, 1] = plan_y
    segments[:, 1, 2] = z_end
    return segments


def format_session_summary(
    session: TrajectorySession,
    *,
    notes: dict[str, str] | None = None,
) -> str:
    from ..common.plotting import format_session_legend_label

    sid = format_session_legend_label(session.session_id, notes)
    lines = [sid]

    def _pivot_line(label: str, fit: MagnetFit) -> None:
        if not fit.is_valid:
            return
        sig = fit.upstream_sigma_mm
        dist = fit.upstream_mm
        if np.isfinite(sig) and sig > 0:
            lines.append(f"  {label} pivot {dist:.0f} ± {sig:.0f} mm upstream")
        else:
            lines.append(f"  {label} pivot {dist:.0f} mm upstream")

    _pivot_line("X scan", session.magnet_x)
    _pivot_line("Y scan", session.magnet_y)

    lines.append(
        f"  align IC1 ({session.align_ic1_x:+.1f}, {session.align_ic1_y:+.1f}) mm",
    )
    lines.append(
        f"  align IC2 ({session.align_ic2_x:+.1f}, {session.align_ic2_y:+.1f}) mm",
    )

    def _iso_line(label: str, iso: IsoFit | None) -> None:
        if iso is None or not iso.is_valid:
            return
        sig = iso.downstream_sigma_mm
        if np.isfinite(sig) and sig > 0:
            lines.append(
                f"  {label} iso {iso.downstream_mm:.0f} ± {sig:.0f} mm down",
            )
        else:
            lines.append(f"  {label} iso {iso.downstream_mm:.0f} mm down")

    _iso_line("X", session.iso_x)
    _iso_line("Y", session.iso_y)
    return "\n".join(lines)
