"""visPy renderer for stacked IC audio waveforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_player_data import FS_HZ, WaveformRenderChannel

_BG = "#1a1a1a"
_FG = "#c9d1d9"
_ACCENT = (0.0, 212 / 255, 170 / 255, 0.9)
_CURSOR_INACTIVE = (58 / 255, 63 / 255, 71 / 255, 0.5)


@dataclass
class _WaveformRow:
    viewbox: object
    fill: object
    wave: object
    cursor: object
    y_lo: float
    y_hi: float


class AudioWaveformScene:
    """Stacked waveform envelopes with a shared playback cursor."""

    def __init__(self, canvas) -> None:
        self._canvas = canvas
        self._grid = self._new_grid()
        self._rows: list[_WaveformRow] = []
        self._duration = 0.0
        self._selected_index = 0
        self._status_node: object | None = None

    def _new_grid(self):
        return self._canvas.central_widget.add_grid(spacing=0)

    @property
    def duration(self) -> float:
        return self._duration

    def clear(self) -> None:
        for row in self._rows:
            row.viewbox.parent = None
        self._rows.clear()
        if self._grid is not None:
            self._grid.parent = None
        self._grid = self._new_grid()
        if self._status_node is not None:
            self._status_node.parent = None
            self._status_node = None
        self._duration = 0.0

    def show_status(self, message: str) -> None:
        from vispy import scene

        self.clear()
        view = self._grid.add_view(row=0, col=0, row_span=1, col_span=1)
        _lock_view_camera(view, scene.PanZoomCamera(aspect=None))
        view.camera.set_range(x=(-1.0, 1.0), y=(-1.0, 1.0))
        text = scene.Text(
            message,
            color=_FG,
            font_size=14,
            pos=(0, 0),
            anchor_x="center",
            anchor_y="center",
            parent=view.scene,
        )
        self._status_node = text
        self._canvas.update()

    def set_render_channels(
        self,
        channels: list[WaveformRenderChannel],
        *,
        selected_index: int = 0,
        cursor_time: float = 0.0,
    ) -> None:
        from vispy import scene

        self.clear()
        if not channels:
            self.show_status("No channels selected")
            return

        self._selected_index = max(0, min(selected_index, len(channels) - 1))
        max_samples = max(len(ch.signal) for ch in channels)
        self._duration = max_samples / FS_HZ

        for i, channel in enumerate(channels):
            view = self._grid.add_view(row=i, col=0, row_span=1, col_span=1)
            camera = scene.PanZoomCamera(aspect=None)
            _lock_view_camera(view, camera)

            fill_rgba = _hex_to_rgba(channel.color, alpha=0.28)
            line_rgba = _hex_to_rgba(channel.color, alpha=0.95)
            mesh_pos, mesh_faces = _mesh_from_envelope_poly(channel.envelope_poly)
            fill = scene.visuals.Mesh(
                vertices=mesh_pos,
                faces=mesh_faces,
                color=fill_rgba,
                parent=view.scene,
            )
            fill.set_gl_state("translucent", depth_test=False)
            wave = scene.visuals.Line(
                pos=channel.line_pos,
                color=line_rgba,
                width=1.2,
                antialias=True,
                parent=view.scene,
            )

            cursor = scene.visuals.Line(
                pos=_cursor_segment(0.0, channel.y_lo, channel.y_hi),
                color=_ACCENT,
                width=2.0,
                antialias=True,
                parent=view.scene,
            )

            if i == len(channels) - 1:
                _add_time_axis(view, self._duration, channel.y_lo)

            view.camera.set_range(
                x=(0.0, self._duration),
                y=(channel.y_lo, channel.y_hi),
            )
            self._rows.append(
                _WaveformRow(
                    viewbox=view,
                    fill=fill,
                    wave=wave,
                    cursor=cursor,
                    y_lo=channel.y_lo,
                    y_hi=channel.y_hi,
                )
            )

        self.set_cursor(cursor_time)
        self._canvas.update()

    def set_selected_index(self, index: int) -> None:
        if not self._rows:
            return
        self._selected_index = max(0, min(index, len(self._rows) - 1))
        self._update_cursor_colors()

    def set_cursor(self, time_s: float) -> None:
        if not self._rows:
            return
        time_s = max(0.0, min(float(time_s), self._duration))
        for row in self._rows:
            row.cursor.set_data(pos=_cursor_segment(time_s, row.y_lo, row.y_hi))
        self._canvas.update()

    def time_at_canvas_pos(self, pos: tuple[float, float]) -> float | None:
        """Map canvas pixel x to time. Camera is locked to [0, duration]."""
        if not self._rows or self._duration <= 0.0:
            return None
        width = float(getattr(self._canvas, "size", (0, 0))[0])
        if width <= 0.0:
            return None
        return max(0.0, min(self._duration * float(pos[0]) / width, self._duration))

    def _update_cursor_colors(self) -> None:
        for i, row in enumerate(self._rows):
            color = _ACCENT if i == self._selected_index else _CURSOR_INACTIVE
            width = 2.0 if i == self._selected_index else 1.0
            row.cursor.set_data(color=color, width=width)


def _lock_view_camera(view, camera) -> None:
    view.bgcolor = _BG
    camera.interactive = False
    view.camera = camera


def _add_time_axis(view, duration: float, y_lo: float) -> None:
    from vispy import scene

    scene.Axis(
        pos=np.array([[0.0, y_lo], [duration, y_lo]], dtype=np.float32),
        domain=(0.0, duration),
        tick_direction=(0.0, -1.0),
        axis_color=(0.45, 0.48, 0.52, 1.0),
        tick_color=(0.45, 0.48, 0.52, 1.0),
        text_color=_FG,
        font_size=8,
        parent=view.scene,
    )


def _cursor_segment(time_s: float, y_lo: float, y_hi: float) -> np.ndarray:
    return np.array([[time_s, y_lo], [time_s, y_hi]], dtype=np.float32)


def _hex_to_rgba(hex_color: str, *, alpha: float) -> tuple[float, float, float, float]:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return (0.7, 0.7, 0.7, alpha)
    r = int(color[0:2], 16) / 255.0
    g = int(color[2:4], 16) / 255.0
    b = int(color[4:6], 16) / 255.0
    return (r, g, b, alpha)


def _mesh_from_envelope_poly(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangle strip covering the min/max envelope (no concave triangulation)."""
    if len(poly) < 4:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
    n = len(poly) // 2
    t = poly[:n, 0]
    y_max = poly[:n, 1]
    y_min = poly[n:, 1][::-1]
    pos = np.zeros((n * 2, 3), dtype=np.float32)
    pos[0::2, 0] = t
    pos[0::2, 1] = y_min
    pos[1::2, 0] = t
    pos[1::2, 1] = y_max
    faces = np.empty((2 * max(n - 1, 0), 3), dtype=np.uint32)
    if n > 1:
        idx = np.arange(n - 1, dtype=np.uint32)
        faces[0::2, 0] = idx * 2
        faces[0::2, 1] = idx * 2 + 1
        faces[0::2, 2] = idx * 2 + 2
        faces[1::2, 0] = idx * 2 + 1
        faces[1::2, 1] = idx * 2 + 3
        faces[1::2, 2] = idx * 2 + 2
    return pos, faces


def _envelope_polygon(
    t: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> np.ndarray:
    """Closed polygon tracing max forward then min backward (tests)."""
    from .audio_player_data import _envelope_polygon as polygon_from_data

    return polygon_from_data(t, y_min, y_max)
