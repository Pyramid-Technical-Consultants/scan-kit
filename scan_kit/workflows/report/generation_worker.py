"""Background worker for PDF report generation."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from .builder import build_report_pdf
from .types import ReportConfig


class ReportGenerationWorker(QObject):
    """Build a PDF report on a worker thread."""

    progress = Signal(int, str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, config: ReportConfig) -> None:
        super().__init__()
        self._config = config

    @Slot()
    def run(self) -> None:
        try:
            output_path, _results = build_report_pdf(
                self._config,
                progress=self.progress.emit,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(str(output_path))
