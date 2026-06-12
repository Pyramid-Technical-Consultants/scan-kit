"""Matplotlib GUI backend setup for frozen view subprocesses."""

from __future__ import annotations

import os
import sys

from .app_icon import prepare_qt_app_identity

_initialized = False


def init_matplotlib_for_views() -> None:
    """Select the QtAgg backend after ``QApplication`` exists.

    Matplotlib 3.x refuses to switch to QtAgg once the headless/Agg backend
    is active.  PyInstaller view subprocesses must therefore create
    ``QApplication`` before importing ``matplotlib.pyplot``.
    """
    global _initialized
    if _initialized:
        return

    if not getattr(sys, "frozen", False):
        _initialized = True
        return

    prepare_qt_app_identity()
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        QApplication(sys.argv)

    # Runtime hook may have set this too early; clear before selecting backend.
    os.environ.pop("MPLBACKEND", None)

    import matplotlib

    matplotlib.use("QtAgg", force=True)
    _initialized = True
