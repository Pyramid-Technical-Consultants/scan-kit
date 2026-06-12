"""In-app debug log panel for the scan-kit launcher."""

from __future__ import annotations

import logging
import sys
import threading
import warnings
from datetime import datetime
from typing import Callable, TextIO

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_MAX_LINES = 5000
_LOG_FORMAT = "%(levelname)s [%(name)s] %(message)s"


def format_log_line(*, level: str, source: str, message: str, now: datetime | None = None) -> str:
    """Build a single display line for the debug panel."""
    ts = (now or datetime.now()).strftime("%H:%M:%S")
    source_part = source.strip() or "app"
    text = message.rstrip("\n")
    return f"{ts} [{level}] [{source_part}] {text}"


class _LogBridge(QObject):
    message = Signal(str, str, str)


class QtLogHandler(logging.Handler):
    """Thread-safe logging handler that forwards records to a debug panel."""

    def __init__(self, emit_fn: Callable[[str, str, str], None]) -> None:
        super().__init__()
        self._emit_fn = emit_fn

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._emit_fn(record.levelname, record.name, msg)
        except Exception:
            self.handleError(record)


class _StreamTee(TextIO):
    """Mirror writes to the original stream and the debug panel."""

    def __init__(
        self,
        original: TextIO,
        *,
        level: str,
        source: str,
        emit_fn: Callable[[str, str, str], None],
    ) -> None:
        self._original = original
        self._level = level
        self._source = source
        self._emit_fn = emit_fn
        self._buffer = ""

    def write(self, text: str) -> int:
        if text and text != "\n":
            self._original.write(text)
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line:
                    self._emit_fn(self._level, self._source, line)
        elif text == "\n":
            self._original.write(text)
            if self._buffer:
                self._emit_fn(self._level, self._source, self._buffer)
                self._buffer = ""
        return len(text)

    def flush(self) -> None:
        self._original.flush()
        if self._buffer:
            self._emit_fn(self._level, self._source, self._buffer)
            self._buffer = ""

    def isatty(self) -> bool:
        try:
            return self._original.isatty()
        except Exception:
            return False

    @property
    def encoding(self) -> str | None:
        return getattr(self._original, "encoding", None)

    @property
    def errors(self) -> str | None:
        return getattr(self._original, "errors", None)


class DebugLogPanel(QWidget):
    """Scrollable log view for launcher and subprocess diagnostics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bridge = _LogBridge()
        self._bridge.message.connect(self._append_message_slot, Qt.ConnectionType.QueuedConnection)
        self._line_count = 0
        self._installed = False
        self._stderr_tee: _StreamTee | None = None
        self._orig_excepthook = sys.excepthook
        self._orig_showwarning = warnings.showwarning
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel(
            "Python log output from the launcher and child view processes "
            "(logging, warnings, stdout, and stderr)."
        )
        title.setWordWrap(True)
        header.addWidget(title, stretch=1)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(self.copy_to_clipboard)
        header.addWidget(copy_btn)
        header.addWidget(clear_btn)
        root.addLayout(header)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._text.setFont(mono)
        root.addWidget(self._text, stretch=1)

    def append(self, level: str, source: str, message: str) -> None:
        """Thread-safe append; safe to call from worker threads."""
        self._bridge.message.emit(level, source, message)

    @Slot(str, str, str)
    def _append_message_slot(self, level: str, source: str, message: str) -> None:
        line = format_log_line(level=level, source=source, message=message)
        self._text.appendPlainText(line)
        self._line_count += 1
        if self._line_count > _MAX_LINES:
            doc = self._text.document()
            excess = self._line_count - _MAX_LINES
            cursor = self._text.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(
                cursor.MoveOperation.Down,
                cursor.MoveMode.KeepAnchor,
                excess,
            )
            cursor.removeSelectedText()
            cursor.deleteChar()
            self._line_count = _MAX_LINES
        self._text.moveCursor(self._text.textCursor().MoveOperation.End)

    def clear(self) -> None:
        self._text.clear()
        self._line_count = 0

    def copy_to_clipboard(self) -> None:
        QGuiApplication.clipboard().setText(self._text.toPlainText())

    def install_logging(self, *, level: int = logging.INFO) -> None:
        """Attach root logging, warnings, excepthook, and stderr mirroring."""
        if self._installed:
            return
        self._installed = True

        handler = QtLogHandler(self.append)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))

        root = logging.getLogger()
        root.setLevel(level)
        root.addHandler(handler)

        if sys.stderr is not None:
            self._stderr_tee = _StreamTee(
                sys.stderr,
                level="STDERR",
                source="stderr",
                emit_fn=self.append,
            )
            sys.stderr = self._stderr_tee

        def _excepthook(exc_type, exc, tb) -> None:
            import traceback

            lines = traceback.format_exception(exc_type, exc, tb)
            for line in lines:
                for part in line.rstrip("\n").splitlines():
                    self.append("ERROR", "uncaught", part)
            if self._orig_excepthook is not None:
                self._orig_excepthook(exc_type, exc, tb)

        sys.excepthook = _excepthook

        def _showwarning(message, category, filename, lineno, file=None, line=None) -> None:
            text = warnings.formatwarning(message, category, filename, lineno, line)
            for part in text.rstrip("\n").splitlines():
                self.append("WARNING", "warnings", part)
            if self._orig_showwarning is not None:
                self._orig_showwarning(message, category, filename, lineno, file, line)

        warnings.showwarning = _showwarning

        self.append("INFO", "launcher", "Debug log started")

    def attach_subprocess_stderr(self, proc, source: str) -> None:
        """Drain a subprocess stderr stream into the panel."""
        stderr = proc.stderr
        if stderr is None:
            return

        def _reader() -> None:
            try:
                for raw in stderr:
                    text = raw.decode(errors="replace").rstrip("\r\n")
                    if text:
                        self.append("STDERR", source, text)
            except Exception:
                pass

        threading.Thread(target=_reader, name=f"debug-log-{source}", daemon=True).start()
