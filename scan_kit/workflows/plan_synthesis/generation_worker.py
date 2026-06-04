"""Background worker for plan synthesis CSV generation."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from .base import PlanTemplate


class PlanGenerationWorker(QObject):
    """Generate an input_map DataFrame on a worker thread."""

    progress = Signal(int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, template: PlanTemplate, params: dict[str, Any]) -> None:
        super().__init__()
        self._template = template
        self._params = params

    @Slot()
    def run(self) -> None:
        try:
            df = self._template.generate(
                self._params,
                progress=self.progress.emit,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(df)
