"""Background worker for PDF report generation."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from .builder import build_report_pdf
from .types import ReportConfig


class ReportGenerationWorker(QObject):
    """Build a PDF report on a worker thread."""

    progress = Signal(int, str)
    finished = Signal(str, str)
    failed = Signal(str)

    def __init__(self, config: ReportConfig) -> None:
        super().__init__()
        self._config = config

    @Slot()
    def run(self) -> None:
        try:
            output_path, results = build_report_pdf(
                self._config,
                progress=self.progress.emit,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        skipped = [r for r in results if not r.success]
        summary = "\n".join(
            f"{result.display_name}: {result.skip_reason or 'skipped'}"
            for result in skipped
        )
        self.finished.emit(str(output_path), summary)
