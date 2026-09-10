"""visPy renderer for stacked IC audio waveforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_player_data import FS_HZ, WaveformRenderChannel

_BG = "#1a1a1a"
_FG = "#c9d1d9"
_ACCENT = (0.0, 212 / 255, 170 / 255, 0.9)
_CURSOR_INACTIVE = (58 / 255, 63 / 255, 71 / 255, 0.5)
_LABEL_MARGIN_FRAC = 0.10


@dataclass
class _WaveformRow:
    viewbox: object
    envelope: object
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
        x_margin = _label_margin(self._duration)

        for i, channel in enumerate(channels):
            view = self._grid.add_view(row=i, col=0, row_span=1, col_span=1)
            camera = scene.PanZoomCamera(aspect=None)
            _lock_view_camera(view, camera)

            rgba = _hex_to_rgba(channel.color, alpha=0.75)
            envelope = scene.visuals.Mesh(
                pos=channel.mesh_pos,
                faces=channel.mesh_faces,
                color=rgba,
                parent=view.scene,
            )
            envelope.set_gl_state("translucent", depth_test=False, cull_face=False)

            y_mid = (channel.y_lo + channel.y_hi) * 0.5
            cursor = scene.visuals.Line(
                pos=_cursor_segment(0.0, channel.y_lo, channel.y_hi),
                color=_CURSOR_INACTIVE,
                width=1.5,
                antialias=True,
                parent=view.scene,
            )

            scene.Text(
                channel.label,
                color=_FG,
                font_size=10,
                pos=(-x_margin * 0.98, y_mid),
                anchor_x="right",
                anchor_y="center",
                parent=view.scene,
            )

            if i == len(channels) - 1:
                _add_time_axis(view, self._duration, channel.y_lo)

            view.camera.set_range(
                x=(-x_margin, self._duration),
                y=(channel.y_lo, channel.y_hi),
            )
            self._rows.append(
                _WaveformRow(
                    viewbox=view,
                    envelope=envelope,
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
            row.cursor.set_data(
                pos=_cursor_segment(time_s, row.y_lo, row.y_hi),
            )
        self._update_cursor_colors()
        self._canvas.update()

    def pick_at_canvas_pos(
        self,
        pos: tuple[float, float],
    ) -> tuple[int, float] | None:
        row_index = self.row_index_at_canvas_pos(pos)
        if row_index is None:
            return None
        time_s = self.time_at_canvas_pos(pos, row_index)
        if time_s is None:
            return None
        return row_index, time_s

    def row_index_at_canvas_pos(self, pos: tuple[float, float]) -> int | None:
        if not self._rows:
            return None
        y = float(pos[1])
        for i, row in enumerate(self._rows):
            vb = row.viewbox
            geom = vb.geometry
            if geom is None:
                continue
            top = geom[1]
            bottom = geom[1] + geom[3]
            if top <= y <= bottom:
                return i
        return None

    def time_at_canvas_pos(
        self,
        pos: tuple[float, float],
        row_index: int,
    ) -> float | None:
        if not self._rows:
            return None
        if row_index < 0 or row_index >= len(self._rows):
            return None
        row = self._rows[row_index]
        tr = row.viewbox.scene.node_transform(row.viewbox.scene)
        mapped = tr.imap(pos)
        if mapped is None:
            return None
        return max(0.0, min(float(mapped[0]), self._duration))

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


def _label_margin(duration: float) -> float:
    return max(duration * _LABEL_MARGIN_FRAC, 0.25)


def _hex_to_rgba(hex_color: str, *, alpha: float) -> tuple[float, float, float, float]:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return (0.7, 0.7, 0.7, alpha)
    r = int(color[0:2], 16) / 255.0
    g = int(color[2:4], 16) / 255.0
    b = int(color[4:6], 16) / 255.0
    return (r, g, b, alpha)


def _envelope_polygon(
    t: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> np.ndarray:
    """Closed polygon tracing max forward then min backward (tests)."""
    n = len(t)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    upper = np.column_stack([t, y_max])
    lower = np.column_stack([t[::-1], y_min[::-1]])
    return np.vstack([upper, lower]).astype(np.float32)


def _envelope_mesh(
    t: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangle mesh for envelope fill (tests)."""
    from .audio_player_data import _envelope_mesh as mesh_from_data

    return mesh_from_data(t, y_min, y_max)
