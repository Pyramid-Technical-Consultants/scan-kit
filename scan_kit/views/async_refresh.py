"""Debounce UI churn and offload heavy session loads from the Qt main thread."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal


class DebouncedBackgroundTask(QObject):
    """Debounce rapid triggers, then run *fn* on a worker thread."""

    finished = Signal(int, object)

    def __init__(
        self,
        *,
        debounce_ms: int = 80,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._debounce_ms = debounce_ms
        self._generation = 0
        self._pending_fn: Callable[[], Any] | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._start_worker)

    def schedule(self, fn: Callable[[], Any]) -> int:
        self._generation += 1
        gen = self._generation
        self._pending_fn = fn
        self._timer.start(self._debounce_ms)
        return gen

    @property
    def generation(self) -> int:
        return self._generation

    def _start_worker(self) -> None:
        gen = self._generation
        fn = self._pending_fn
        if fn is None:
            return

        def worker() -> None:
            try:
                result = fn()
            except Exception:
                result = None
            self.finished.emit(gen, result)

        threading.Thread(
            target=worker,
            daemon=True,
            name="plot-refresh",
        ).start()
