"""Keep the pytest suite headless (no blocking matplotlib or Qt windows)."""

from __future__ import annotations

import os

# Must be set before matplotlib is imported anywhere in the test process.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _headless_matplotlib():
    """Prevent ``plt.show()`` from opening a blocking GUI window during tests."""
    captured: list[plt.Figure] = []

    def _capture_show(*args, **kwargs) -> None:
        del args, kwargs
        for num in plt.get_fignums():
            fig = plt.figure(num)
            if fig not in captured:
                captured.append(fig)

    real_show = plt.show
    plt.show = _capture_show
    try:
        yield
    finally:
        plt.show = real_show
        plt.close("all")


@pytest.fixture(autouse=True)
def _headless_qt_windows():
    """Layout-only ``QWidget.show()`` calls without flashing real windows."""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QWidget
    except ImportError:
        yield
        return

    real_show = QWidget.show

    def _show_offscreen(self, *args, **kwargs):
        self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        return real_show(self, *args, **kwargs)

    QWidget.show = _show_offscreen
    try:
        yield
    finally:
        QWidget.show = real_show
