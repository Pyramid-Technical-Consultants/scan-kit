"""Audio Explorer visPy scenes (waveform stack + live FFT)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_player_data import (
    FS_HZ,
    SPECTRUM_DB_FLOOR,
    SPECTRUM_FMAX_HZ,
    WaveformRenderChannel,
)
from .vispy_plot import (
    ACCENT_RGBA,
    FG,
    add_fill_mesh,
    add_line,
    add_locked_xy_plot,
    add_shared_x_axis,
    add_status_text,
    envelope_mesh_geometry,
    hex_to_rgba,
    lock_panzoom,
    map_canvas_x_to_data,
    vertical_segments,
)

_CURSOR_INACTIVE = (58 / 255, 63 / 255, 71 / 255, 0.5)
_PEAK_MARK = (0.72, 0.76, 0.80, 0.35)
_SPECTRUM_Y = (SPECTRUM_DB_FLOOR, 5.0)


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
        self.clear()
        view = self._grid.add_view(row=0, col=0, row_span=1, col_span=1)
        lock_panzoom(view)
        view.camera.set_range(x=(-1.0, 1.0), y=(-1.0, 1.0))
        self._status_node = add_status_text(view, message)
        self._canvas.update()

    def set_render_channels(
        self,
        channels: list[WaveformRenderChannel],
        *,
        selected_index: int = 0,
        cursor_time: float = 0.0,
    ) -> None:
        self.clear()
        if not channels:
            self.show_status("No channels selected")
            return

        self._selected_index = max(0, min(selected_index, len(channels) - 1))
        max_samples = max(len(ch.signal) for ch in channels)
        self._duration = max_samples / FS_HZ

        for i, channel in enumerate(channels):
            plot = add_locked_xy_plot(
                self._grid,
                row=i,
                x_range=(0.0, self._duration),
                y_range=(channel.y_lo, channel.y_hi),
            )
            view = plot.view
            verts, faces = envelope_mesh_geometry(channel.envelope_poly)
            fill = add_fill_mesh(
                view.scene, verts, faces, hex_to_rgba(channel.color, alpha=0.28),
            )
            wave = add_line(
                view.scene,
                channel.line_pos,
                color=hex_to_rgba(channel.color, alpha=0.95),
                width=1.2,
            )
            cursor = add_line(
                view.scene,
                vertical_segments(0.0, channel.y_lo, channel.y_hi),
                color=ACCENT_RGBA,
                width=2.0,
                order=1,
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

        add_shared_x_axis(
            self._grid, row=len(channels), view=self._rows[-1].viewbox,
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
                pos=vertical_segments(time_s, row.y_lo, row.y_hi),
            )
        self._canvas.update()

    def time_at_canvas_pos(self, pos: tuple[float, float]) -> float | None:
        if not self._rows or self._duration <= 0.0:
            return None
        return map_canvas_x_to_data(
            self._rows[0].viewbox,
            pos[0],
            0.0,
            self._duration,
            canvas_width=float(getattr(self._canvas, "size", (0, 0))[0]),
        )

    def _update_cursor_colors(self) -> None:
        for i, row in enumerate(self._rows):
            color = ACCENT_RGBA if i == self._selected_index else _CURSOR_INACTIVE
            width = 2.0 if i == self._selected_index else 1.0
            row.cursor.set_data(color=color, width=width)


class AudioSpectrumScene:
    """Live frequency-magnitude plot for the sample window at the playhead."""

    def __init__(self, canvas) -> None:
        self._canvas = canvas
        self._view = None
        self._line = None
        self._peak_marks = None
        self._ensure_view()

    def _ensure_view(self) -> None:
        if self._view is not None:
            return
        grid = self._canvas.central_widget.add_grid(spacing=0)
        plot = add_locked_xy_plot(
            grid,
            x_range=(0.0, SPECTRUM_FMAX_HZ),
            y_range=_SPECTRUM_Y,
            x_axis=True,
            right_gutter=True,
        )
        self._line = add_line(
            plot.view.scene, color=FG, width=1.4, visible=False,
        )
        self._peak_marks = add_line(
            plot.view.scene,
            color=_PEAK_MARK,
            width=1.0,
            connect="segments",
            order=1,
            visible=False,
        )
        self._view = plot.view
        self._xaxis = plot.xaxis
        self._yaxis = plot.yaxis

    def set_spectrum(
        self,
        freqs: np.ndarray,
        db: np.ndarray,
        *,
        color: str = "#c9d1d9",
        peaks: np.ndarray | None = None,
    ) -> None:
        self._ensure_view()
        n = min(int(np.asarray(freqs).size), int(np.asarray(db).size))
        if n == 0:
            self._line.visible = False
        else:
            freqs = np.asarray(freqs)[:n]
            db = np.asarray(db)[:n]
            mask = freqs <= SPECTRUM_FMAX_HZ + 1e-9
            pos = np.column_stack((freqs[mask], db[mask])).astype(np.float32)
            if len(pos) == 0:
                self._line.visible = False
            else:
                self._line.visible = True
                self._line.set_data(
                    pos=pos, color=hex_to_rgba(color, alpha=0.95),
                )
        marks = _peak_marker_segments(peaks)
        self._peak_marks.visible = len(marks) >= 2
        if self._peak_marks.visible:
            self._peak_marks.set_data(pos=marks)
        self._canvas.update()


def _peak_marker_segments(peaks: np.ndarray | None) -> np.ndarray:
    if peaks is None or len(peaks) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    hz = np.asarray(peaks, dtype=np.float32)
    hz = hz[(hz >= 0.0) & (hz <= SPECTRUM_FMAX_HZ + 1e-9)]
    return vertical_segments(hz, SPECTRUM_DB_FLOOR, _SPECTRUM_Y[1])


def _envelope_polygon(
    t: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> np.ndarray:
    """Closed polygon tracing max forward then min backward (tests)."""
    from .audio_player_data import _envelope_polygon as polygon_from_data

    return polygon_from_data(t, y_min, y_max)
