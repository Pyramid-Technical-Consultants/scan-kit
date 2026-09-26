"""Hairline progress bar laid over the top edge of a view."""

from __future__ import annotations

import time

from PySide6.QtCore import QEvent, QObject, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPalette
from PySide6.QtWidgets import QWidget

_FRAME_MS = 16
_SWEEP_S = 1.4
_EASE = 0.22  # share of the gap to the target closed each frame
_FADE_IN, _FADE_OUT = 0.25, 0.07  # opacity per frame


class ProgressLine(QWidget):
    """A thin line across the top of *host* while the user waits on something.

    :meth:`busy` sweeps for work of unknown length, :meth:`set_progress` fills to a
    fraction, and :meth:`done` completes the fill and fades out. Idle, it paints
    nothing and takes no clicks, so the view underneath is untouched.
    """

    def __init__(self, host: QWidget, *, thickness: int = 2) -> None:
        super().__init__(host)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFixedHeight(thickness)
        self._target: float | None = None  # None sweeps
        self._shown = 0.0
        self._opacity = 0.0
        self._active = False
        self._t0 = time.monotonic()
        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_MS)
        self._timer.timeout.connect(self._tick)
        host.installEventFilter(self)
        self._fit()

    @property
    def active(self) -> bool:
        return self._active

    def busy(self) -> None:
        if self._target is not None or self._opacity == 0.0:
            self._t0 = time.monotonic()
        self._target = None
        self._start()

    def set_progress(self, fraction: float) -> None:
        f = min(max(float(fraction), 0.0), 1.0)
        if self._target is None or not self._active:
            self._shown = 0.0
        elif f < self._shown:
            self._shown = f
        self._target = f
        self._start()

    def done(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._target is not None:
            self._target = 1.0
        self._animate()

    def _start(self) -> None:
        self._active = True
        self.raise_()
        self._animate()

    def _animate(self) -> None:
        # start() on a running QTimer restarts it, so frequent callers would starve the frames.
        if not self._timer.isActive():
            self._timer.start()

    def _tick(self) -> None:
        step = _FADE_IN if self._active else -_FADE_OUT
        self._opacity = min(max(self._opacity + step, 0.0), 1.0)
        if self._target is not None:
            self._shown += (self._target - self._shown) * _EASE
        if not self._active and self._opacity == 0.0:
            self._timer.stop()
            self._shown = 0.0
        elif self._active and self._target is not None and self._opacity == 1.0 and abs(self._target - self._shown) < 1e-3:
            self._shown = self._target
            self._timer.stop()
        self.update()

    def _fit(self) -> None:
        self.setGeometry(0, 0, self.parentWidget().width(), self.height())

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Resize:
            self._fit()
        return False

    def paintEvent(self, _event) -> None:
        if self._opacity <= 0.0:
            return
        w, h = float(self.width()), float(self.height())
        accent = self.palette().color(QPalette.ColorRole.Highlight)
        track = QColor(accent)
        track.setAlphaF(0.15)
        p = QPainter(self)
        p.setOpacity(self._opacity)
        p.fillRect(QRectF(0.0, 0.0, w, h), track)
        if self._target is None:
            phase = ((time.monotonic() - self._t0) / _SWEEP_S) % 1.0
            phase = phase * phase * (3.0 - 2.0 * phase)
            seg = 0.3 * w
            x = -seg + phase * (w + seg)
            clear = QColor(accent)
            clear.setAlphaF(0.0)
            grad = QLinearGradient(x, 0.0, x + seg, 0.0)
            grad.setColorAt(0.0, clear)
            grad.setColorAt(0.5, accent)
            grad.setColorAt(1.0, clear)
            p.fillRect(QRectF(x, 0.0, seg, h), grad)
        else:
            p.fillRect(QRectF(0.0, 0.0, self._shown * w, h), accent)
        p.end()
