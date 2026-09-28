"""Full-height color bar with major and minor ticks, shared by the dose views."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .dose_volume_fill import colormap_samples, ink_rgb, zero_rgb
from .vispy_plot import BG

_AXIS_PAD_X = 8
_AXIS_BAR_W = 16
# Fixed so tick text never shoves the view sideways; the ×10 offset rides in the title.
_AXIS_WIDTH = 124
_EXP_DIGITS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

def _tick_labels(values: np.ndarray) -> tuple[list[str], str]:
    """Short tick text plus a shared power-of-ten offset when the span is tiny or huge."""
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return [], ""
    peak = float(np.max(np.abs(vals)))
    scale = 1.0
    offset = ""
    if peak > 0.0 and np.isfinite(peak):
        exp = int(np.floor(np.log10(peak)))
        if exp < -2 or exp > 3:
            scale = 10.0 ** exp
            offset = "×10" + str(exp).translate(_EXP_DIGITS)
    labels = []
    for value in vals:
        shown = float(value) / scale
        if abs(shown) < 1e-8:
            shown = 0.0
        labels.append(f"{shown:.4g}".replace("-", "−"))
    return labels, offset


def _minor_parts(step: float) -> int:
    if step <= 0.0 or not np.isfinite(step):
        return 5
    leading = step / 10.0 ** np.floor(np.log10(step))
    if abs(leading - 2.0) < 0.05 or abs(leading - 2.5) < 0.05:
        return 4
    return 5


def _minor_ticks(majors: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if majors.size < 2:
        return np.array([], dtype=float)
    out: list[float] = []

    def _fill(start: float, step: float, *, forward: bool) -> None:
        parts = _minor_parts(abs(step))
        delta = abs(step) / parts
        value = start
        for _ in range(32):
            value = value + delta if forward else value - delta
            if value <= lo or value >= hi:
                return
            out.append(value)

    for left, right in zip(majors[:-1], majors[1:]):
        step = float(right - left)
        parts = _minor_parts(step)
        for k in range(1, parts):
            value = float(left) + step * k / parts
            if lo < value < hi:
                out.append(value)
    _fill(float(majors[0]), float(majors[1] - majors[0]), forward=False)
    _fill(float(majors[-1]), float(majors[-1] - majors[-2]), forward=True)
    return np.asarray(out, dtype=float)


def color_axis_ticks(
    lo: float, hi: float, *, nbins: int = 6,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Major ticks, minor ticks, major labels, and the scientific offset.

    ``majors[0]`` and ``majors[-1]`` are always the exact ends of the bar.
    """
    lo_f = float(lo)
    hi_f = float(hi)
    empty = np.array([], dtype=float)
    if not np.isfinite(lo_f) or not np.isfinite(hi_f):
        return empty, empty, [], ""
    if hi_f < lo_f:
        lo_f, hi_f = hi_f, lo_f
    span = hi_f - lo_f
    if span <= 0.0:
        return np.array([lo_f]), empty, [], ""
    from matplotlib import ticker

    locator = ticker.MaxNLocator(
        nbins=max(2, int(nbins)),
        steps=[1, 2, 2.5, 5, 10],
        min_n_ticks=2,
    )
    nice = np.asarray(locator.tick_values(lo_f, hi_f), dtype=float)
    pad = span * 1e-6
    nice = nice[(nice >= lo_f - pad) & (nice <= hi_f + pad)]
    if lo_f < 0.0 < hi_f and not np.any(np.abs(nice) <= pad):
        nice = np.sort(np.append(nice, 0.0))
    minors = _minor_ticks(nice, lo_f, hi_f) if nice.size >= 2 else np.array([], dtype=float)
    inner = nice[(nice > lo_f + pad) & (nice < hi_f - pad)]
    majors = np.concatenate([[lo_f], inner, [hi_f]])
    labels, offset = _tick_labels(majors)
    return majors, minors, labels, offset


class ColorAxis(QWidget):
    """Full-height color bar with major ticks, minor ticks, and a unit title."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._name = ""
        self._lo = float("nan")
        self._hi = float("nan")
        self._title = ""
        self._images: dict[tuple[str, int], QImage] = {}
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setFixedWidth(_AXIS_WIDTH)

    def sizeHint(self) -> QSize:
        return QSize(_AXIS_WIDTH, 480)

    def set_scale(self, name: str, lo: float, hi: float, title: str) -> None:
        lo_f = float(lo)
        hi_f = float(hi)
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f
        same = (
            name == self._name
            and title == self._title
            and np.isfinite(self._lo)
            and abs(lo_f - self._lo) <= 1e-9 * max(1.0, abs(lo_f))
            and abs(hi_f - self._hi) <= 1e-9 * max(1.0, abs(hi_f))
        )
        if same:
            return
        self._name = name
        self._lo = lo_f
        self._hi = hi_f
        self._title = title
        self.update()

    def _bar_image(self, height: int) -> QImage | None:
        if height < 2 or not self._name:
            return None
        key = (self._name, height)
        cached = self._images.get(key)
        if cached is not None:
            return cached
        rgb = np.clip(colormap_samples(self._name, height)[::-1], 0.0, 1.0)
        rgba = np.zeros((height, 1, 4), dtype=np.uint8)
        rgba[:, 0, :3] = (rgb * 255.0).astype(np.uint8)
        rgba[:, 0, 3] = 255
        rgba = np.ascontiguousarray(np.repeat(rgba, _AXIS_BAR_W, axis=1))
        image = QImage(
            rgba.data, _AXIS_BAR_W, height, int(rgba.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()
        self._images[key] = image
        if len(self._images) > 24:
            self._images.pop(next(iter(self._images)))
        return image

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        fm = painter.fontMetrics()
        lo, hi = self._lo, self._hi
        if not self._name or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            painter.fillRect(self.rect(), QColor(BG))
            painter.end()
            return
        # Painted as part of the view, so it shares the canvas zero color,
        # with a dark glass panel so the ticks read on any background.
        painter.fillRect(self.rect(), QColor.fromRgbF(*zero_rgb(self._name, lo, hi)))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(12, 14, 18, 170))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(2, 2, -2, -2), 6, 6)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        color = QColor.fromRgbF(*ink_rgb((0.0, 0.0, 0.0)))
        nbins = max(2, min(8, int(self.height() / 52)))
        majors, minors, labels, offset = color_axis_ticks(lo, hi, nbins=nbins)
        top = fm.height() // 2 + 4
        bottom = self.height() - (fm.height() // 2 + 4)
        if bottom - top < 8:
            painter.end()
            return
        bar = QRectF(_AXIS_PAD_X, top, _AXIS_BAR_W, bottom - top)
        image = self._bar_image(max(2, int(bar.height())))
        if image is not None:
            painter.drawImage(bar, image)
        pen = QPen(color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(bar)
        spine = bar.right()

        def y_of(value: float) -> float:
            return top + (hi - value) / (hi - lo) * (bottom - top)

        for value in minors:
            y = y_of(float(value))
            painter.drawLine(int(spine), int(round(y)), int(spine) + 4, int(round(y)))
        shown_y: list[float] = []
        # Ends first so the top and bottom labels always win a collision.
        order = [0, len(majors) - 1, *range(1, len(majors) - 1)]
        for idx in order:
            value = float(majors[idx])
            y = y_of(value)
            if any(abs(y - prev) < fm.height() for prev in shown_y):
                painter.drawLine(int(spine), int(round(y)), int(spine) + 5, int(round(y)))
                continue
            length = 9 if abs(value) <= (hi - lo) * 1e-6 else 7
            painter.drawLine(int(spine), int(round(y)), int(spine) + length, int(round(y)))
            painter.drawText(
                int(spine) + length + 4,
                int(round(y + fm.ascent() / 2 - 1)),
                labels[idx],
            )
            shown_y.append(y)
        title = f"{self._title}  {offset}".strip() if offset else self._title
        if title:
            title_x = self.width() - 4 - fm.height()
            painter.save()
            painter.translate(title_x + fm.height() / 2, (top + bottom) / 2)
            painter.rotate(-90)
            painter.drawText(
                QRectF(-bar.height() / 2, -fm.height() / 2, bar.height(), fm.height()),
                Qt.AlignmentFlag.AlignCenter,
                title,
            )
            painter.restore()
        painter.end()
