"""visPy 3D renderer for IC beam trajectories."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..common.ic_trajectory import IC1_Z_MM, IC2_Z_MM
from ..common.plotting import DEFAULT_SESSION_COLORS
from .trajectory_catalog import TrajectoryConfig
from .trajectory_data import (
    TrajectorySession,
    plan_segments_3d,
    segment_z_extent,
    spot_segments_3d,
)


def _beam_to_scene(segments: np.ndarray) -> np.ndarray:
    """Map beam coords ``(lateral_x, lateral_y, z_downstream)`` to vispy ``(x, y, z)``.

    Beam downstream is vispy +X (horizontal); lateral x is +Y; lateral y is +Z depth.
    Matches the 2D trajectory panels where downstream is horizontal and lateral is vertical.
    """
    return np.stack(
        [segments[..., 2], segments[..., 0], segments[..., 1]],
        axis=-1,
    )


def _scene_point(x_lateral: float, y_lateral: float, z_beam: float) -> np.ndarray:
    return np.array([z_beam, x_lateral, y_lateral], dtype=float)


def _segments_to_line_pos(segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ``(n, 2, 3)`` beam segments to vispy line positions and edge indices."""
    if segments.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 2), dtype=int)
    scene = _beam_to_scene(segments)
    n = scene.shape[0]
    pos = scene.reshape(n * 2, 3)
    connect = np.arange(n * 2, dtype=np.int32).reshape(n, 2)
    return pos, connect


def _energy_colors(energy: np.ndarray) -> np.ndarray:
    """Per-spot RGBA from energy (viridis-like via matplotlib cmap without pyplot)."""
    from matplotlib import cm

    finite = np.isfinite(energy)
    if not finite.any():
        return np.tile([0.5, 0.5, 0.5, 0.35], (energy.size, 1))
    e = energy[finite]
    lo, hi = float(np.min(e)), float(np.max(e))
    norm = np.zeros(energy.size, dtype=float)
    if hi > lo:
        norm[finite] = (energy[finite] - lo) / (hi - lo)
    else:
        norm[finite] = 0.5
    rgba = np.asarray(cm.viridis(norm), dtype=np.float32).copy()
    rgba[~finite, 3] = 0.0
    rgba[finite, 3] = 0.45
    return rgba


def _session_color_rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, alpha)


class TrajectoryScene:
    """Build and update a vispy scene for trajectory data."""

    def __init__(self, canvas) -> None:
        from vispy import scene

        self._canvas = canvas
        self._view = canvas.central_widget.add_view()
        self._view.camera = scene.cameras.TurntableCamera(
            fov=45,
            distance=2500,
            center=(IC1_Z_MM / 2, 0.0, 0.0),
        )
        self._view.camera.set_range()
        self._nodes: list = []

        # Beam axis (+X downstream from IC2); kept outside _nodes so clear() does not remove it.
        self._axis = scene.visuals.Line(
            pos=np.array(
                [
                    [IC2_Z_MM - 500, 0.0, 0.0],
                    [IC1_Z_MM + 500, 0.0, 0.0],
                ],
                dtype=float,
            ),
            color=(0.4, 0.4, 0.4, 0.6),
            width=1,
            parent=self._view.scene,
        )

    def clear(self) -> None:
        for node in self._nodes:
            node.parent = None
        self._nodes.clear()

    def render(
        self,
        sessions: dict[str, TrajectorySession],
        config: TrajectoryConfig,
        session_ids: Sequence[str],
        colors: Sequence[str],
    ) -> None:
        from vispy import scene

        self.clear()

        if not sessions:
            text = scene.Text(
                "No trajectory data loaded",
                color="white",
                font_size=16,
                pos=(IC1_Z_MM / 2, 0.0, 0.0),
                parent=self._view.scene,
            )
            self._nodes.append(text)
            self._canvas.update()
            return

        z_min = IC2_Z_MM - config.extend_upstream_mm
        z_max = IC1_Z_MM + config.extend_downstream_mm
        pivot_z_vals: list[float] = []
        lateral_span = 120.0

        def _add_pivot_marker(z_beam: float, color: str, symbol: str) -> None:
            pivot_z_vals.append(z_beam)
            marker = scene.visuals.Markers(
                pos=np.array([_scene_point(0.0, 0.0, z_beam)], dtype=float),
                face_color=_session_color_rgba(color, 0.95),
                size=10,
                symbol=symbol,
                parent=self._view.scene,
            )
            self._nodes.append(marker)

        for sid, color in zip(session_ids, colors):
            session = sessions.get(sid)
            if session is None:
                continue

            rgba = _session_color_rgba(color, 0.12)

            if config.show_spots:
                segments = spot_segments_3d(
                    session,
                    extend_upstream_mm=config.extend_upstream_mm,
                    extend_downstream_mm=config.extend_downstream_mm,
                )
                pos, connect = _segments_to_line_pos(segments)
                if pos.size:
                    z_start, z_end = segment_z_extent(
                        session,
                        extend_upstream_mm=config.extend_upstream_mm,
                        extend_downstream_mm=config.extend_downstream_mm,
                    )
                    z_min = min(z_min, z_start)
                    z_max = max(z_max, z_end)
                    lateral_span = max(
                        lateral_span,
                        float(np.nanmax(np.abs(pos[:, 1]))),
                        float(np.nanmax(np.abs(pos[:, 2]))),
                    )
                    if session.energy is not None and np.isfinite(session.energy).any():
                        line_colors = _energy_colors(session.energy)
                        seg_colors = np.repeat(line_colors, 2, axis=0)
                        line = scene.visuals.Line(
                            pos=pos,
                            connect=connect,
                            color=seg_colors,
                            width=1,
                            parent=self._view.scene,
                        )
                    else:
                        line = scene.visuals.Line(
                            pos=pos,
                            connect=connect,
                            color=rgba,
                            width=1,
                            parent=self._view.scene,
                        )
                    self._nodes.append(line)

            if config.show_plan:
                plan_segs = plan_segments_3d(
                    session,
                    extend_upstream_mm=config.extend_upstream_mm,
                    extend_downstream_mm=config.extend_downstream_mm,
                )
                if plan_segs is not None:
                    pos, connect = _segments_to_line_pos(plan_segs)
                    plan_rgba = _session_color_rgba(color, 0.35)
                    line = scene.visuals.Line(
                        pos=pos,
                        connect=connect,
                        color=plan_rgba,
                        width=1.5,
                        parent=self._view.scene,
                    )
                    self._nodes.append(line)

            if config.show_pivot_markers:
                if session.magnet_x.is_valid and np.isfinite(session.magnet_x.z_pivot):
                    _add_pivot_marker(session.magnet_x.z_pivot, color, "o")
                    label = scene.Text(
                        "X scan",
                        pos=_scene_point(0.0, 0.0, session.magnet_x.z_pivot),
                        color=_session_color_rgba(color, 0.9),
                        font_size=9,
                        parent=self._view.scene,
                    )
                    self._nodes.append(label)
                if session.magnet_y.is_valid and np.isfinite(session.magnet_y.z_pivot):
                    z_py = session.magnet_y.z_pivot
                    if not (
                        session.magnet_x.is_valid
                        and np.isfinite(session.magnet_x.z_pivot)
                        and abs(z_py - session.magnet_x.z_pivot) < 1.0
                    ):
                        _add_pivot_marker(z_py, color, "s")
                        label = scene.Text(
                            "Y scan",
                            pos=_scene_point(0.0, 0.0, z_py),
                            color=_session_color_rgba(color, 0.9),
                            font_size=9,
                            parent=self._view.scene,
                        )
                        self._nodes.append(label)
                    elif not session.magnet_x.is_valid:
                        _add_pivot_marker(z_py, color, "s")

            if config.show_iso_planes:
                iso_rgba = _session_color_rgba(color, 0.15)
                if session.iso_x is not None and session.iso_x.is_valid:
                    plane = _iso_plane_mesh(session.iso_x.z_iso, span=80.0, color=iso_rgba)
                    plane.parent = self._view.scene
                    self._nodes.append(plane)
                if session.iso_y is not None and session.iso_y.is_valid:
                    z_iy = session.iso_y.z_iso
                    if session.iso_x is None or not session.iso_x.is_valid or abs(
                        z_iy - session.iso_x.z_iso,
                    ) > 5.0:
                        plane = _iso_plane_mesh(z_iy, span=80.0, color=iso_rgba)
                        plane.parent = self._view.scene
                        self._nodes.append(plane)

        if config.show_ic_planes:
            for z_plane, label in ((IC2_Z_MM, "IC2"), (IC1_Z_MM, "IC1")):
                grid = _ic_plane_grid(z_plane, span=60.0)
                grid.parent = self._view.scene
                self._nodes.append(grid)
                label_vis = scene.Text(
                    label,
                    pos=_scene_point(0.0, 0.0, z_plane),
                    color=(0.5, 0.5, 0.5, 1.0),
                    font_size=10,
                    parent=self._view.scene,
                )
                self._nodes.append(label_vis)

        if pivot_z_vals:
            z_min = min(pivot_z_vals) - config.pivot_margin_mm

        self._view.camera.center = ((z_min + z_max) / 2, 0.0, 0.0)
        pad = max(lateral_span, 60.0)
        self._view.camera.set_range(
            x=(z_min, z_max),
            y=(-pad, pad),
            z=(-pad, pad),
        )
        self._canvas.update()


def _ic_plane_grid(z_beam: float, span: float) -> object:
    from vispy import scene

    s = span
    corners = np.array(
        [
            [z_beam, -s, -s],
            [z_beam, s, -s],
            [z_beam, s, s],
            [z_beam, -s, s],
            [z_beam, -s, -s],
        ],
        dtype=float,
    )
    return scene.visuals.Line(
        pos=corners,
        color=(0.55, 0.55, 0.55, 0.85),
        width=1.5,
        method="gl",
        connect="strip",
    )


def _iso_plane_mesh(z_beam: float, span: float, color: tuple[float, float, float, float]) -> object:
    from vispy import scene

    s = span
    vertices = np.array(
        [
            [z_beam, -s, -s],
            [z_beam, s, -s],
            [z_beam, s, s],
            [z_beam, -s, s],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return scene.visuals.Mesh(
        vertices=vertices,
        faces=faces,
        color=color,
        shading="flat",
    )


def default_session_colors(n: int) -> list[str]:
    return list(DEFAULT_SESSION_COLORS[:n])
