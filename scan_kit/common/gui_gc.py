"""Garbage collection on the GUI thread only.

Python collects in whichever thread happens to allocate when a threshold trips. A vispy
canvas left in a reference cycle then frees its GL context from a loader thread, and
Qt kills the process ("Cannot make QOpenGLContext current in a different thread").
"""

from __future__ import annotations

import gc

INTERVAL_MS = 100


def collect_due() -> int:
    """Collect the generations whose thresholds have tripped, as Python would; the objects freed."""
    t0, t1, t2 = gc.get_threshold()
    c0, c1, c2 = gc.get_count()
    if c0 <= t0:
        return 0
    return gc.collect(2 if c2 > t2 and c1 > t1 else 1 if c1 > t1 else 0)


def collect_on_gui_thread(app):
    """Turn off automatic collection and run :func:`collect_due` from *app*'s thread; the timer."""
    from PySide6.QtCore import QTimer

    gc.disable()
    timer = QTimer(app)
    timer.setInterval(INTERVAL_MS)
    timer.timeout.connect(collect_due)
    timer.start()
    return timer
