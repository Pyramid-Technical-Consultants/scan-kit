"""The hairline progress bar tracks its host and vanishes once the work is done."""

from __future__ import annotations

import time


def test_progress_line_fades_out_when_done(qapp) -> None:
    from PySide6.QtWidgets import QWidget

    from scan_kit.common.progress_line import ProgressLine

    host = QWidget()
    host.resize(200, 50)
    line = ProgressLine(host)
    host.show()
    host.resize(300, 50)
    assert line.width() == 300 and not line.active
    line.done()
    assert not line._timer.isActive()
    line.busy()
    deadline = time.monotonic() + 2.0
    while line._shown != 0.4 and time.monotonic() < deadline:
        line.set_progress(0.4)  # as often as a busy caller would
        qapp.processEvents()
        time.sleep(0.005)
    assert line.active and line._shown == 0.4
    line.done()
    deadline = time.monotonic() + 2.0
    while line._timer.isActive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    assert not line.active and line._opacity == 0.0 and not line._timer.isActive()
    host.deleteLater()
