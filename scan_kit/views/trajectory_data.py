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
from ..common.ic_trajectory import IC1_Z_MM, IC2_Z_MM, IC_SEP_MM, ic_alignment_offsets
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
        return combined_pivot_z(self.magnet_x, self.magnet_y)

    @property
    def iso_z(self) -> float:
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


def spot_segments_3d(
    session: TrajectorySession,
    *,
    extend_upstream_mm: float,
    extend_downstream_mm: float,
) -> np.ndarray:
    """Return ``(n_spots, 2, 3)`` line endpoints in mm (x, y, z downstream from IC2)."""
    n = session.n_spots
    if n == 0:
        return np.empty((0, 2, 3), dtype=float)

    z_up = IC2_Z_MM - extend_upstream_mm
    z_dn = IC1_Z_MM + extend_downstream_mm

    sx = (session.ic1_x - session.ic2_x) / IC_SEP_MM
    sy = (session.ic1_y - session.ic2_y) / IC_SEP_MM

    x_up = session.ic2_x + sx * (z_up - IC2_Z_MM)
    y_up = session.ic2_y + sy * (z_up - IC2_Z_MM)
    x_dn = session.ic1_x + sx * (z_dn - IC1_Z_MM)
    y_dn = session.ic1_y + sy * (z_dn - IC1_Z_MM)

    segments = np.empty((n, 2, 3), dtype=float)
    segments[:, 0, 0] = x_up
    segments[:, 0, 1] = y_up
    segments[:, 0, 2] = z_up
    segments[:, 1, 0] = x_dn
    segments[:, 1, 1] = y_dn
    segments[:, 1, 2] = z_dn
    return segments


def plan_segments_3d(
    session: TrajectorySession,
    *,
    extend_upstream_mm: float,
    extend_downstream_mm: float,
) -> np.ndarray | None:
    """Per-spot plan rays in 3D, or None when plan or fits are unavailable."""
    if session.plan_x is None or session.plan_y is None:
        return None
    z_pivot = session.pivot_z
    z_iso = session.iso_z
    if not np.isfinite(z_pivot) or not np.isfinite(z_iso) or abs(z_iso - z_pivot) < 1e-3:
        return None

    plan_x = session.plan_x
    plan_y = session.plan_y
    ok = np.isfinite(plan_x) & np.isfinite(plan_y)
    if not np.any(ok):
        return None

    plan_x = plan_x[ok]
    plan_y = plan_y[ok]
    n = plan_x.size

    z_up = IC2_Z_MM - extend_upstream_mm
    z_dn = IC1_Z_MM + extend_downstream_mm
    scale_up = (z_up - z_pivot) / (z_iso - z_pivot)
    scale_dn = (z_dn - z_pivot) / (z_iso - z_pivot)

    segments = np.empty((n, 2, 3), dtype=float)
    segments[:, 0, 0] = plan_x * scale_up
    segments[:, 0, 1] = plan_y * scale_up
    segments[:, 0, 2] = z_up
    segments[:, 1, 0] = plan_x * scale_dn
    segments[:, 1, 1] = plan_y * scale_dn
    segments[:, 1, 2] = z_dn
    return segments


def format_session_summary(
    session: TrajectorySession,
    *,
    notes: dict[str, str] | None = None,
) -> str:
    from ..common.plotting import format_session_legend_label

    sid = format_session_legend_label(session.session_id, notes)
    lines = [sid]
    if session.magnet_x.is_valid:
        sig = session.magnet_x.upstream_sigma_mm
        dist = session.magnet_x.upstream_mm
        if np.isfinite(sig) and sig > 0:
            lines.append(f"  pivot {dist:.0f} ± {sig:.0f} mm upstream")
        else:
            lines.append(f"  pivot {dist:.0f} mm upstream")
    lines.append(
        f"  align IC1 ({session.align_ic1_x:+.1f}, {session.align_ic1_y:+.1f}) mm",
    )
    lines.append(
        f"  align IC2 ({session.align_ic2_x:+.1f}, {session.align_ic2_y:+.1f}) mm",
    )
    z_iso = session.iso_z
    if np.isfinite(z_iso):
        lines.append(f"  iso plane z = {z_iso:.0f} mm downstream")
    return "\n".join(lines)
