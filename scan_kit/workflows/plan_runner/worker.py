"""Background worker for plan runner RCI operations."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from .service import PlanRunnerService
from .session_packager import download_session_zip


class PlanRunnerWorker(QObject):
    """Run RCI I/O on a worker thread (connect, upload, controls, download)."""

    connected = Signal(object)
    connection_failed = Signal(str)
    disconnected = Signal()
    upload_done = Signal(str)
    upload_failed = Signal(str)
    operation_failed = Signal(str)
    status_updated = Signal(object)
    status_error = Signal(str)
    download_done = Signal(str)
    download_failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._service = PlanRunnerService()
        self._poll_timer: QTimer | None = None

    def _ensure_timer(self) -> QTimer:
        if self._poll_timer is None:
            timer = QTimer(self)
            timer.setInterval(250)
            timer.timeout.connect(self._poll_status)
            self._poll_timer = timer
        return self._poll_timer

    @Slot(str)
    def connect_host(self, host: str) -> None:
        try:
            info = self._service.connect(host)
            self._ensure_timer().start()
            self.connected.emit(info)
            self.status_updated.emit(info.get("status") or self._service.read_status())
        except Exception as exc:
            self.connection_failed.emit(str(exc))

    @Slot()
    def disconnect_host(self) -> None:
        if self._poll_timer is not None:
            self._poll_timer.stop()
        self._service.disconnect()
        self.disconnected.emit()

    @Slot(str)
    def upload_plan(self, csv_path: str) -> None:
        try:
            target = self._service.upload_plan(Path(csv_path))
            self.upload_done.emit(target)
            self.status_updated.emit(self._service.read_status())
        except Exception as exc:
            self.upload_failed.emit(str(exc))

    @Slot()
    def start_run(self) -> None:
        self._run_control(self._service.start)

    @Slot()
    def pause_run(self) -> None:
        self._run_control(self._service.pause)

    @Slot()
    def stop_run(self) -> None:
        self._run_control(self._service.stop)

    @Slot()
    def reset_run(self) -> None:
        self._run_control(self._service.reset)

    @Slot(str, str)
    def download_session(self, remote_session_path: str, zip_path: str) -> None:
        try:
            host = self._service.host
            if not host:
                raise RuntimeError("not connected")
            path = download_session_zip(host, remote_session_path, Path(zip_path))
            self.download_done.emit(str(path))
        except Exception as exc:
            self.download_failed.emit(str(exc))

    def _run_control(self, fn) -> None:  # type: ignore[no-untyped-def]
        try:
            fn()
            self.status_updated.emit(self._service.read_status())
        except Exception as exc:
            self.operation_failed.emit(str(exc))

    def _poll_status(self) -> None:
        if not self._service.connected:
            return
        try:
            self.status_updated.emit(self._service.read_status())
        except Exception as exc:
            self.status_error.emit(str(exc))
