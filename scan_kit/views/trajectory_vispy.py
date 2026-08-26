"""visPy 3D renderer for IC beam trajectories."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..common.scan_magnet_model import (
    D2_650_POLE_BLOCK_DEPTH_MM,
    D2_650_POLE_GAP_X_MM,
    D2_650_POLE_GAP_Y_MM,
    D2_650_POLE_LENGTH_X_MM,
    D2_650_POLE_LENGTH_Y_MM,
    PoleBoxSpec,
    beam_lateral_mm,
    dual_dipole_pole_boxes,
    measured_lateral_xy,
    reference_on_axis_trace,
)
from ..common.ic_trajectory import (
    IC1_Z_MM,
    IC2_Z_MM,
    IC128_25_HALF_SPAN_MM,
    ic_strip_midplane_z,
    ic_strip_planes,
)
from ..common.plotting import DEFAULT_SESSION_COLORS
from .trajectory_catalog import (
    ENERGY_RAY_ALPHA,
    ISO_GRID_MAJOR_STEP_MM,
    ISO_GRID_STEP_MM,
    ISO_PLANE_HALF_HEIGHT_MM,
    ISO_PLANE_HALF_WIDTH_MM,
    PLANE_SPOT_MARKER_SIZE,
    PLANE_SPOT_MARKER_Z_EPSILON_MM,
    PLAN_RAY_ALPHA,
    SPOT_RAY_ALPHA,
    TrajectoryConfig,
)
from .trajectory_data import (
    TrajectorySession,
    plan_lateral_on_plane,
    plan_traces_3d,
    segment_z_extent,
    spot_traces_3d,
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
    """Flatten ``(n, 2, 3)`` or ``(n, k, 3)`` beam paths to vispy line positions."""
    if segments.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 2), dtype=int)
    if segments.ndim != 3:
        raise ValueError(f"expected (n, k, 3) array, got shape {segments.shape}")
    scene = _beam_to_scene(segments)
    n, k, _ = scene.shape
    pos = scene.reshape(n * k, 3)
    connect = np.empty((n * (k - 1), 2), dtype=np.int32)
    row = 0
    for i in range(n):
        base = i * k
        for j in range(k - 1):
            connect[row, 0] = base + j
            connect[row, 1] = base + j + 1
            row += 1
    return pos, connect


def _energy_colors(energy: np.ndarray) -> np.ndarray:
    """Per-spot RGBA from energy (viridis-like via matplotlib cmap without pyplot)."""
    from matplotlib import cm

    finite = np.isfinite(energy)
    if not finite.any():
        return np.tile([0.5, 0.5, 0.5, ENERGY_RAY_ALPHA], (energy.size, 1))
    e = energy[finite]
    lo, hi = float(np.min(e)), float(np.max(e))
    norm = np.zeros(energy.size, dtype=float)
    if hi > lo:
        norm[finite] = (energy[finite] - lo) / (hi - lo)
    else:
        norm[finite] = 0.5
    rgba = np.asarray(cm.viridis(norm), dtype=np.float32).copy()
    rgba[~finite, 3] = 0.0
    rgba[finite, 3] = ENERGY_RAY_ALPHA
    return rgba


def _trace_vertex_colors(energy: np.ndarray, knots_per_trace: int) -> np.ndarray:
    """Per-vertex RGBA for polylines: one energy color repeated per knot."""
    line_colors = _energy_colors(energy)
    return np.repeat(line_colors, knots_per_trace, axis=0)


def _session_color_rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, alpha)


def _marker_face_rgba(line_rgba: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Spot marker RGB matches the ray; markers are drawn fully opaque."""
    r, g, b, _a = line_rgba
    return (r, g, b, 1.0)


def _energy_marker_colors(energy: np.ndarray) -> np.ndarray:
    rgba = _energy_colors(energy)
    rgba[:, 3] = 1.0
    return rgba


def _make_opaque_disc_marker(
    parent,
    pos: np.ndarray,
    face_color,
    *,
    size: float,
    symbol: str = "disc",
) -> object:
    """Flat opaque markers (no GL blend / antialias fringe)."""
    from vispy import scene

    marker = scene.visuals.Markers(
        pos=pos,
        face_color=face_color,
        size=size,
        symbol=symbol,
        edge_width=0,
        edge_color=(0.0, 0.0, 0.0, 0.0),
        spherical=False,
        method="instanced",
        antialias=0,
        alpha=1.0,
        parent=parent,
    )
    # Default Markers use src_alpha blending + soft edges, which moirés on dense grids.
    marker.set_gl_state(preset="opaque")
    return marker


def _plane_spot_marker_positions(
    lateral_x: np.ndarray,
    lateral_y: np.ndarray,
    z_beam: float,
) -> np.ndarray:
    """Scene positions for spot markers on a beam-normal plane."""
    lx = np.asarray(lateral_x, dtype=float)
    ly = np.asarray(lateral_y, dtype=float)
    ok = np.isfinite(lx) & np.isfinite(ly)
    if not np.any(ok):
        return np.empty((0, 3), dtype=float)
    n = int(np.count_nonzero(ok))
    z = np.full(n, z_beam, dtype=float)
    return np.column_stack([z, lx[ok], ly[ok]])


def _add_plane_spot_markers(
    parent,
    nodes: list,
    lateral_x: np.ndarray,
    lateral_y: np.ndarray,
    z_beam: float,
    line_rgba: tuple[float, float, float, float],
    *,
    energy: np.ndarray | None = None,
) -> None:
    lx = np.asarray(lateral_x, dtype=float)
    ly = np.asarray(lateral_y, dtype=float)
    ok = np.isfinite(lx) & np.isfinite(ly)
    if not np.any(ok):
        return
    n = int(np.count_nonzero(ok))
    z_draw = z_beam + PLANE_SPOT_MARKER_Z_EPSILON_MM
    pos = np.column_stack(
        [
            np.full(n, z_draw, dtype=float),
            lx[ok],
            ly[ok],
        ],
    )
    if energy is not None and np.isfinite(np.asarray(energy)).any():
        face_color = _energy_marker_colors(np.asarray(energy, dtype=float)[ok])
    else:
        face_color = _marker_face_rgba(line_rgba)
    marker = _make_opaque_disc_marker(
        parent,
        pos,
        face_color,
        size=PLANE_SPOT_MARKER_SIZE,
    )
    nodes.append(marker)


def _measured_lateral_on_plane(
    session: TrajectorySession,
    z_beam: float,
) -> tuple[np.ndarray, np.ndarray]:
    z_px = session.magnet_x.z_pivot if session.magnet_x.is_valid else float("nan")
    z_py = session.magnet_y.z_pivot if session.magnet_y.is_valid else float("nan")
    if session.magnet_x.is_valid and session.magnet_y.is_valid:
        return measured_lateral_xy(
            np.full(session.n_spots, z_beam),
            session.ic2_x,
            session.ic1_x,
            session.ic2_y,
            session.ic1_y,
            z_px,
            z_py,
        )
    return (
        beam_lateral_mm(session.ic2_x, session.ic1_x, z_beam),
        beam_lateral_mm(session.ic2_y, session.ic1_y, z_beam),
    )


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
        upstream_extent_z: list[float] = []
        lateral_span = IC128_25_HALF_SPAN_MM
        show_magnet_labels = len(session_ids) == 1

        def _track_upstream_extent(session: TrajectorySession) -> None:
            if session.magnet_x.is_valid:
                upstream_extent_z.append(session.magnet_x.z_pivot)
            if session.magnet_y.is_valid:
                upstream_extent_z.append(session.magnet_y.z_pivot)

        def _add_pivot_marker(z_beam: float, color: str, symbol: str) -> None:
            marker = _make_opaque_disc_marker(
                self._view.scene,
                np.array([_scene_point(0.0, 0.0, z_beam)], dtype=float),
                _session_color_rgba(color, 1.0),
                size=10,
                symbol=symbol,
            )
            self._nodes.append(marker)

        for sid, color in zip(session_ids, colors):
            session = sessions.get(sid)
            if session is None:
                continue

            rgba = _session_color_rgba(color, SPOT_RAY_ALPHA)
            plan_rgba = _session_color_rgba(color, PLAN_RAY_ALPHA)

            if config.show_spot_lines or config.show_spot_markers:
                _track_upstream_extent(session)

            if config.show_spot_lines:
                traces = spot_traces_3d(
                    session,
                    extend_upstream_mm=config.extend_upstream_mm,
                    extend_downstream_mm=config.extend_downstream_mm,
                )
                pos, connect = _segments_to_line_pos(traces)
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
                        knots_per_trace = traces.shape[1]
                        seg_colors = _trace_vertex_colors(session.energy, knots_per_trace)
                        if seg_colors.shape[0] != pos.shape[0]:
                            # Mismatch should not happen; fall back to uniform color.
                            seg_colors = None
                        line = scene.visuals.Line(
                            pos=pos,
                            connect=connect,
                            color=seg_colors if seg_colors is not None else rgba,
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

            if config.show_plan_lines:
                plan_traces = plan_traces_3d(
                    session,
                    extend_upstream_mm=config.extend_upstream_mm,
                    extend_downstream_mm=config.extend_downstream_mm,
                )
                if plan_traces is not None:
                    pos, connect = _segments_to_line_pos(plan_traces)
                    line = scene.visuals.Line(
                        pos=pos,
                        connect=connect,
                        color=plan_rgba,
                        width=1.5,
                        parent=self._view.scene,
                    )
                    self._nodes.append(line)

            if config.show_plan_markers:
                if config.show_ic_planes:
                    for chamber_z in (IC2_Z_MM, IC1_Z_MM):
                        z_mid = ic_strip_midplane_z(chamber_z)
                        plan_xy = plan_lateral_on_plane(session, z_mid)
                        if plan_xy is not None:
                            px, py = plan_xy
                            _add_plane_spot_markers(
                                self._view.scene,
                                self._nodes,
                                px,
                                py,
                                z_mid,
                                plan_rgba,
                            )
                if config.show_iso_planes and np.isfinite(session.isocenter_z):
                    z_iso = session.isocenter_z
                    plan_xy = plan_lateral_on_plane(session, z_iso)
                    if plan_xy is not None:
                        px, py = plan_xy
                        _add_plane_spot_markers(
                            self._view.scene,
                            self._nodes,
                            px,
                            py,
                            z_iso,
                            plan_rgba,
                        )

            if config.show_pivot_markers:
                _track_upstream_extent(session)
                geom = session.dipole_geometry
                use_gap_center = (
                    config.show_magnet_gaps
                    and geom is not None
                    and geom.is_valid
                )
                if session.magnet_x.is_valid:
                    z_x = (
                        geom.z_magnet_center_x
                        if use_gap_center
                        else session.magnet_x.z_pivot
                    )
                    if np.isfinite(z_x):
                        _add_pivot_marker(z_x, color, "o")
                        if show_magnet_labels:
                            label = scene.Text(
                                "X magnet",
                                pos=_scene_point(0.0, 0.0, z_x),
                                color=_session_color_rgba(color, 0.9),
                                font_size=9,
                                parent=self._view.scene,
                            )
                            self._nodes.append(label)
                if session.magnet_y.is_valid:
                    z_y = (
                        geom.z_magnet_center_y
                        if use_gap_center
                        else session.magnet_y.z_pivot
                    )
                    if np.isfinite(z_y):
                        _add_pivot_marker(z_y, color, "s")
                        if show_magnet_labels:
                            label = scene.Text(
                                "Y magnet",
                                pos=_scene_point(0.0, 0.0, z_y),
                                color=_session_color_rgba(color, 0.9),
                                font_size=9,
                                parent=self._view.scene,
                            )
                            self._nodes.append(label)

            if config.show_magnet_gaps:
                geom = session.dipole_geometry
                if geom is not None and geom.is_valid:
                    gap_rgba = _session_color_rgba(color, 0.22)
                    for spec in dual_dipole_pole_boxes(geom):
                        box = _pole_box_visual(spec, gap_rgba)
                        box.parent = self._view.scene
                        self._nodes.append(box)
                    z_start, z_end = segment_z_extent(
                        session,
                        extend_upstream_mm=config.extend_upstream_mm,
                        extend_downstream_mm=config.extend_downstream_mm,
                    )
                    ref_trace = reference_on_axis_trace(z_start, z_end, geom)
                    ref_pos, ref_connect = _segments_to_line_pos(ref_trace[np.newaxis, :, :])
                    ref_line = scene.visuals.Line(
                        pos=ref_pos,
                        connect=ref_connect,
                        color=(0.85, 0.85, 0.85, 0.55),
                        width=2,
                        parent=self._view.scene,
                    )
                    self._nodes.append(ref_line)
                    half_x = D2_650_POLE_LENGTH_X_MM / 2.0
                    half_y = D2_650_POLE_LENGTH_Y_MM / 2.0
                    z_min = min(z_min, geom.z_magnet_center_x - half_x, geom.z_magnet_center_y - half_y)
                    z_max = max(z_max, geom.z_magnet_center_x + half_x, geom.z_magnet_center_y + half_y)
                    lateral_span = max(
                        lateral_span,
                        D2_650_POLE_GAP_X_MM / 2.0 + D2_650_POLE_BLOCK_DEPTH_MM,
                        D2_650_POLE_GAP_Y_MM / 2.0 + D2_650_POLE_BLOCK_DEPTH_MM,
                    )

            if config.show_iso_planes and np.isfinite(session.isocenter_z):
                z_iso = session.isocenter_z
                for vis in _iso_plane_visual(
                    z_iso,
                    ISO_PLANE_HALF_WIDTH_MM,
                    ISO_PLANE_HALF_HEIGHT_MM,
                    _session_color_rgba(color, 0.15),
                ):
                    vis.parent = self._view.scene
                    self._nodes.append(vis)
                z_max = max(z_max, z_iso)
                lateral_span = max(
                    lateral_span,
                    ISO_PLANE_HALF_WIDTH_MM,
                    ISO_PLANE_HALF_HEIGHT_MM,
                )
                if config.show_spot_markers:
                    iso_x, iso_y = _measured_lateral_on_plane(session, z_iso)
                    _add_plane_spot_markers(
                        self._view.scene,
                        self._nodes,
                        iso_x,
                        iso_y,
                        z_iso,
                        rgba,
                        energy=session.energy,
                    )

            if config.show_ic_planes and config.show_spot_markers:
                for chamber_z, ic_x, ic_y in (
                    (IC2_Z_MM, session.ic2_x, session.ic2_y),
                    (IC1_Z_MM, session.ic1_x, session.ic1_y),
                ):
                    _add_plane_spot_markers(
                        self._view.scene,
                        self._nodes,
                        ic_x,
                        ic_y,
                        ic_strip_midplane_z(chamber_z),
                        rgba,
                        energy=session.energy,
                    )

        if config.show_ic_planes:
            for plane in ic_strip_planes():
                for vis in _ic_plane_visual(plane.z_mm, IC128_25_HALF_SPAN_MM):
                    vis.parent = self._view.scene
                    self._nodes.append(vis)
                label_vis = scene.Text(
                    plane.label,
                    pos=_scene_point(0.0, IC128_25_HALF_SPAN_MM * 0.85, plane.z_mm),
                    color=(0.5, 0.5, 0.5, 1.0),
                    font_size=10,
                    parent=self._view.scene,
                )
                self._nodes.append(label_vis)

        if upstream_extent_z:
            z_min = min(z_min, min(upstream_extent_z) - config.pivot_margin_mm)

        self._view.camera.center = ((z_min + z_max) / 2, 0.0, 0.0)
        pad = max(lateral_span, IC128_25_HALF_SPAN_MM, ISO_PLANE_HALF_HEIGHT_MM, 60.0)
        self._view.camera.set_range(
            x=(z_min, z_max),
            y=(-pad, pad),
            z=(-pad, pad),
        )
        self._canvas.update()


def _pole_box_visual(
    spec: PoleBoxSpec,
    color: tuple[float, float, float, float],
) -> object:
    from vispy import scene

    edge = (
        min(1.0, color[0] * 1.1),
        min(1.0, color[1] * 1.1),
        min(1.0, color[2] * 1.1),
        min(1.0, color[3] + 0.25),
    )
    # vispy Box: width=X, depth=Y, height=Z (not width, height, depth = X, Y, Z).
    box = scene.visuals.Box(
        width=spec.width_x,
        depth=spec.width_y,
        height=spec.width_z,
        color=color,
        edge_color=edge,
    )
    box.transform = scene.STTransform(
        translate=(spec.center_x, spec.center_y, spec.center_z),
    )
    return box


def _ic_plane_visual(z_beam: float, half_span_mm: float) -> list:
    """IC128-25 active area: square plane in lateral Y/Z at fixed beam depth."""
    from vispy import scene

    s = half_span_mm
    fill = scene.visuals.Mesh(
        vertices=np.array(
            [
                [z_beam, -s, -s],
                [z_beam, s, -s],
                [z_beam, s, s],
                [z_beam, -s, s],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
        color=(0.45, 0.45, 0.45, 0.10),
        shading="flat",
    )
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
    border = scene.visuals.Line(
        pos=corners,
        color=(0.55, 0.55, 0.55, 0.85),
        width=1.5,
        method="gl",
        connect="strip",
    )
    return [fill, border]


def _grid_major_coord(coord_mm: float, major_step_mm: float) -> bool:
    rem = abs(coord_mm % major_step_mm)
    return rem < 0.05 or rem > major_step_mm - 0.05


def _iso_plane_grid_line_sets(
    z_beam: float,
    half_width_mm: float,
    half_height_mm: float,
    *,
    step_mm: float,
    major_step_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Minor and major grid segments on a beam-normal plane (lateral Y/Z)."""
    minor_pos: list[list[float]] = []
    minor_connect: list[list[int]] = []
    major_pos: list[list[float]] = []
    major_connect: list[list[int]] = []

    def _append(
        p0: list[float],
        p1: list[float],
        is_major: bool,
    ) -> None:
        pos = major_pos if is_major else minor_pos
        connect = major_connect if is_major else minor_connect
        base = len(pos)
        pos.extend([p0, p1])
        connect.append([base, base + 1])

    hw, hh = half_width_mm, half_height_mm
    z_vals = np.arange(-hh, hh + step_mm * 0.5, step_mm)
    y_vals = np.arange(-hw, hw + step_mm * 0.5, step_mm)
    for z in z_vals:
        major = _grid_major_coord(float(z), major_step_mm)
        _append(
            [z_beam, -hw, float(z)],
            [z_beam, hw, float(z)],
            major,
        )
    for y in y_vals:
        major = _grid_major_coord(float(y), major_step_mm)
        _append(
            [z_beam, float(y), -hh],
            [z_beam, float(y), hh],
            major,
        )

    def _as_arrays(
        pos: list[list[float]],
        connect: list[list[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not pos:
            return np.empty((0, 3), dtype=float), np.empty((0, 2), dtype=np.int32)
        return np.asarray(pos, dtype=float), np.asarray(connect, dtype=np.int32)

    m_pos, m_conn = _as_arrays(minor_pos, minor_connect)
    maj_pos, maj_conn = _as_arrays(major_pos, major_connect)
    return m_pos, m_conn, maj_pos, maj_conn


def _iso_plane_visual(
    z_beam: float,
    half_width_mm: float,
    half_height_mm: float,
    color: tuple[float, float, float, float],
) -> list:
    """30×40 cm style isocenter rectangle (width × height in lateral Y/Z)."""
    from vispy import scene

    hw, hh = half_width_mm, half_height_mm
    fill = scene.visuals.Mesh(
        vertices=np.array(
            [
                [z_beam, -hw, -hh],
                [z_beam, hw, -hh],
                [z_beam, hw, hh],
                [z_beam, -hw, hh],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
        color=color,
        shading="flat",
    )
    corners = np.array(
        [
            [z_beam, -hw, -hh],
            [z_beam, hw, -hh],
            [z_beam, hw, hh],
            [z_beam, -hw, hh],
            [z_beam, -hw, -hh],
        ],
        dtype=float,
    )
    edge = (
        min(1.0, color[0] * 1.1),
        min(1.0, color[1] * 1.1),
        min(1.0, color[2] * 1.1),
        min(1.0, color[3] + 0.25),
    )
    border = scene.visuals.Line(
        pos=corners,
        color=edge,
        width=1.5,
        method="gl",
        connect="strip",
    )
    visuals: list = [fill, border]

    minor_pos, minor_conn, major_pos, major_conn = _iso_plane_grid_line_sets(
        z_beam,
        hw,
        hh,
        step_mm=ISO_GRID_STEP_MM,
        major_step_mm=ISO_GRID_MAJOR_STEP_MM,
    )
    minor_color = (
        min(1.0, color[0] * 0.9),
        min(1.0, color[1] * 0.9),
        min(1.0, color[2] * 0.9),
        min(1.0, color[3] + 0.08),
    )
    major_color = (
        min(1.0, color[0] * 1.05),
        min(1.0, color[1] * 1.05),
        min(1.0, color[2] * 1.05),
        min(1.0, color[3] + 0.22),
    )
    if minor_pos.size:
        visuals.append(
            scene.visuals.Line(
                pos=minor_pos,
                connect=minor_conn,
                color=minor_color,
                width=1,
                method="gl",
            ),
        )
    if major_pos.size:
        visuals.append(
            scene.visuals.Line(
                pos=major_pos,
                connect=major_conn,
                color=major_color,
                width=2,
                method="gl",
            ),
        )
    return visuals


def default_session_colors(n: int) -> list[str]:
    return list(DEFAULT_SESSION_COLORS[:n])
