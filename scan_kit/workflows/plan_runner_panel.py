"""Plan Runner workflow panel — upload plans to RCI and monitor execution."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QMetaObject, Qt, QThread, Q_ARG
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..common.app_settings import AppSettings
from ..common.qt_widgets import configure_pane_scroll_area, make_pane_scroll_area, set_pane_scroll_widget
from ..igx.rci_paths import (
    COMBINED_START_PERMIT,
    CONTROL_POINT_COUNT,
    CONTROL_POINT_INDEX,
    POINTS_VALID,
    POINT_ENERGY,
    POINT_LAYER_ID,
    POINT_PROGRESS,
    PROGRESS,
    STATE,
)
from .plan_runner.worker import PlanRunnerWorker


class PlanRunnerPanel(QWidget):
    """Upload input_map.csv to RCI, run controls, and optionally download session data."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        app_settings: AppSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self._app_settings = app_settings or AppSettings.load()
        self._worker_thread: QThread | None = None
        self._worker: PlanRunnerWorker | None = None
        self._connected = False
        self._build_ui()
        self._start_worker()

    def _build_ui(self) -> None:
        scroll = make_pane_scroll_area(self)
        configure_pane_scroll_area(scroll)
        content = QWidget()
        layout = QVBoxLayout(content)

        conn_box = QGroupBox("RCI connection")
        conn_form = QFormLayout(conn_box)
        self._host_edit = QLineEdit()
        self._host_edit.setPlaceholderText("192.168.100.184")
        if self._app_settings.last_rci_host:
            self._host_edit.setText(self._app_settings.last_rci_host)
        conn_row = QHBoxLayout()
        conn_row.addWidget(self._host_edit, stretch=1)
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        conn_row.addWidget(self._connect_btn)
        conn_form.addRow("Host", conn_row)
        self._conn_status = QLabel("Not connected")
        conn_form.addRow("Status", self._conn_status)
        layout.addWidget(conn_box)

        plan_box = QGroupBox("Treatment plan")
        plan_form = QFormLayout(plan_box)
        self._plan_path_edit = QLineEdit()
        plan_row = QHBoxLayout()
        plan_row.addWidget(self._plan_path_edit, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_plan)
        plan_row.addWidget(browse_btn)
        plan_form.addRow("input_map.csv", plan_row)
        self._upload_btn = QPushButton("Upload to RCI")
        self._upload_btn.setEnabled(False)
        self._upload_btn.clicked.connect(self._on_upload)
        plan_form.addRow(self._upload_btn)
        layout.addWidget(plan_box)

        ctrl_box = QGroupBox("Run controls")
        ctrl_row = QHBoxLayout(ctrl_box)
        self._start_btn = QPushButton("Start")
        self._pause_btn = QPushButton("Pause")
        self._stop_btn = QPushButton("Stop")
        self._reset_btn = QPushButton("Reset")
        for btn in (self._start_btn, self._pause_btn, self._stop_btn, self._reset_btn):
            btn.setEnabled(False)
            ctrl_row.addWidget(btn)
        self._start_btn.clicked.connect(lambda: self._invoke("start_run"))
        self._pause_btn.clicked.connect(lambda: self._invoke("pause_run"))
        self._stop_btn.clicked.connect(lambda: self._invoke("stop_run"))
        self._reset_btn.clicked.connect(lambda: self._invoke("reset_run"))
        layout.addWidget(ctrl_box)

        progress_box = QGroupBox("Progress")
        progress_form = QFormLayout(progress_box)
        self._state_label = QLabel("—")
        self._progress_label = QLabel("—")
        self._point_label = QLabel("—")
        self._energy_label = QLabel("—")
        self._layer_label = QLabel("—")
        self._permit_label = QLabel("—")
        progress_form.addRow("Controller state", self._state_label)
        progress_form.addRow("Progress", self._progress_label)
        progress_form.addRow("Control point", self._point_label)
        progress_form.addRow("Point energy", self._energy_label)
        progress_form.addRow("Layer id", self._layer_label)
        progress_form.addRow("Start permit", self._permit_label)
        layout.addWidget(progress_box)

        session_box = QGroupBox("Session download")
        session_form = QFormLayout(session_box)
        self._session_path_edit = QLineEdit()
        self._session_path_edit.setPlaceholderText("/root/reports/session/<session_id>")
        session_form.addRow("Device path", self._session_path_edit)
        self._download_btn = QPushButton("Download session zip…")
        self._download_btn.setEnabled(False)
        self._download_btn.clicked.connect(self._on_download_session)
        session_form.addRow(self._download_btn)
        layout.addWidget(session_box)

        layout.addStretch(1)
        set_pane_scroll_widget(scroll, content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _start_worker(self) -> None:
        thread = QThread(self)
        worker = PlanRunnerWorker()
        worker.moveToThread(thread)
        worker.connected.connect(self._on_connected)
        worker.connection_failed.connect(self._on_connection_failed)
        worker.disconnected.connect(self._on_disconnected)
        worker.upload_done.connect(self._on_upload_done)
        worker.upload_failed.connect(self._on_upload_failed)
        worker.operation_failed.connect(self._on_operation_failed)
        worker.status_updated.connect(self._on_status_updated)
        worker.download_done.connect(self._on_download_done)
        worker.download_failed.connect(self._on_download_failed)
        thread.start()
        self._worker_thread = thread
        self._worker = worker

    def _invoke(self, method: str, *args: object) -> None:
        worker = self._worker
        if worker is None:
            return
        if not args:
            QMetaObject.invokeMethod(
                worker,
                method,
                Qt.ConnectionType.QueuedConnection,
            )
            return
        if len(args) == 1 and isinstance(args[0], str):
            QMetaObject.invokeMethod(
                worker,
                method,
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, args[0]),
            )
            return
        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
            QMetaObject.invokeMethod(
                worker,
                method,
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, args[0]),
                Q_ARG(str, args[1]),
            )
            return
        raise TypeError(f"unsupported invoke args for {method}")

    def _on_connect_clicked(self) -> None:
        if self._connected:
            self._connect_btn.setEnabled(False)
            self._invoke("disconnect_host")
            return

        host = self._host_edit.text().strip()
        if not host:
            QMessageBox.warning(self, "Plan Runner", "Enter an RCI host IP or hostname.")
            return
        self._connect_btn.setEnabled(False)
        self._conn_status.setText("Connecting…")
        self._app_settings.last_rci_host = host
        self._app_settings.save()
        self._invoke("connect_host", host)

    def _on_browse_plan(self) -> None:
        start_dir = self._app_settings.last_plan_runner_file_dir or ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input_map.csv",
            start_dir,
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        self._plan_path_edit.setText(path)
        self._remember_plan_dir(path)

    def _remember_plan_dir(self, path: str | Path) -> None:
        try:
            self._app_settings.last_plan_runner_file_dir = str(Path(path).parent)
            self._app_settings.save()
        except OSError:
            pass

    def _on_upload(self) -> None:
        path = self._plan_path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Plan Runner", "Choose an input_map.csv file.")
            return
        self._upload_btn.setEnabled(False)
        self._invoke("upload_plan", path)

    def _on_download_session(self) -> None:
        remote = self._session_path_edit.text().strip()
        if not remote:
            QMessageBox.warning(
                self,
                "Plan Runner",
                "Enter the session folder path on the device.",
            )
            return
        start_dir = self._app_settings.last_plan_runner_file_dir or ""
        zip_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save session zip",
            str(Path(start_dir) / f"{Path(remote).name}.zip"),
            "Zip archives (*.zip)",
        )
        if not zip_path:
            return
        self._remember_plan_dir(zip_path)
        self._download_btn.setEnabled(False)
        self._invoke("download_session", remote, zip_path)

    def _set_run_controls_enabled(self, enabled: bool) -> None:
        for btn in (
            self._start_btn,
            self._pause_btn,
            self._stop_btn,
            self._reset_btn,
        ):
            btn.setEnabled(enabled)

    def _on_connected(self, info: object) -> None:
        self._connected = True
        self._connect_btn.setText("Disconnect")
        self._connect_btn.setEnabled(True)
        version = info.get("version") if isinstance(info, dict) else info
        device = info.get("device_type") if isinstance(info, dict) else None
        label = f"Connected — {device or 'IGX'} {version or ''}".strip()
        self._conn_status.setText(label)
        self._upload_btn.setEnabled(True)
        self._download_btn.setEnabled(True)
        if isinstance(info, dict):
            session_dir = info.get("session_directory")
            if session_dir and not self._session_path_edit.text().strip():
                self._session_path_edit.setPlaceholderText(
                    f"{session_dir}/<session_id>"
                )

    def _on_connection_failed(self, message: str) -> None:
        self._connected = False
        self._connect_btn.setText("Connect")
        self._connect_btn.setEnabled(True)
        self._conn_status.setText("Not connected")
        self._upload_btn.setEnabled(False)
        self._download_btn.setEnabled(False)
        self._set_run_controls_enabled(False)
        QMessageBox.critical(self, "Connection failed", message)

    def _on_disconnected(self) -> None:
        self._connected = False
        self._connect_btn.setText("Connect")
        self._connect_btn.setEnabled(True)
        self._conn_status.setText("Not connected")
        self._upload_btn.setEnabled(False)
        self._download_btn.setEnabled(False)
        self._set_run_controls_enabled(False)

    def _on_upload_done(self, target: str) -> None:
        self._upload_btn.setEnabled(True)
        QMessageBox.information(
            self,
            "Upload complete",
            f"Control points loaded to {target}",
        )

    def _on_upload_failed(self, message: str) -> None:
        self._upload_btn.setEnabled(True)
        QMessageBox.critical(self, "Upload failed", message)

    def _on_operation_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Plan Runner", message)

    def _on_status_updated(self, status: object) -> None:
        if not isinstance(status, dict):
            return
        state = status.get(STATE)
        self._state_label.setText(str(state) if state is not None else "—")

        progress = status.get(PROGRESS)
        point_prog = status.get(POINT_PROGRESS)
        if progress is not None:
            text = f"{progress:.1f}%"
            if point_prog is not None:
                text += f" (point {point_prog:.1f}%)"
            self._progress_label.setText(text)
        else:
            self._progress_label.setText("—")

        idx = status.get(CONTROL_POINT_INDEX)
        count = status.get(CONTROL_POINT_COUNT)
        if idx is not None and count is not None:
            self._point_label.setText(f"{idx} / {count}")
        else:
            self._point_label.setText("—")

        energy = status.get(POINT_ENERGY)
        self._energy_label.setText(str(energy) if energy is not None else "—")
        layer = status.get(POINT_LAYER_ID)
        self._layer_label.setText(str(layer) if layer is not None else "—")
        permit = status.get(COMBINED_START_PERMIT)
        self._permit_label.setText("yes" if permit else "no")

        points_ok = status.get(POINTS_VALID)
        can_run = bool(points_ok) and self._connected
        self._set_run_controls_enabled(can_run)
        if state == "completed":
            self._download_btn.setEnabled(True)

    def _on_download_done(self, path: str) -> None:
        self._download_btn.setEnabled(True)
        QMessageBox.information(self, "Session downloaded", f"Saved to {path}")

    def _on_download_failed(self, message: str) -> None:
        self._download_btn.setEnabled(True)
        QMessageBox.critical(self, "Download failed", message)
