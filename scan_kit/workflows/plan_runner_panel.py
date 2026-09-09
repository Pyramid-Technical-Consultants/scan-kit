"""Plan Runner — operator console for uploading and running RCI plans."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QMetaObject, QRectF, Qt, QThread, Q_ARG
from PySide6.QtGui import QCloseEvent, QFont, QPainter, QPainterPath, QShowEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..common.app_settings import AppSettings
from ..common.qt_widgets import make_pane_scroll_area, set_pane_scroll_widget
from ..igx.http import parse_host
from ..igx.rci_paths import (
    COMBINED_POINTS_OK,
    COMBINED_STATE,
    CONTROL_POINT_COUNT,
    POINT_ENERGY,
    POINT_LAYER_ID,
    POINT_PROGRESS,
    POINTS_VALID,
    PROGRESS,
    READY_PERMIT,
    READY_PERMIT_REASON,
    STATE,
    TIME_ELAPSED,
    TREATMENT_ACTIVE,
)
from .plan_runner.status import (
    coach_message,
    control_enables,
    default_session_zip_path,
    format_elapsed,
    format_energy,
    io_bool,
    io_number,
    io_text,
    point_fraction,
    progress_percent,
    resolve_session_download,
    session_download_hint,
    unwrap_io,
)
from .plan_runner.worker import PlanRunnerWorker

_LEFT_PANE_WIDTH = 456
_LEFT_PANE_MIN = 380
_RIGHT_PANE_MIN = 420

_STATE_COLORS = {
    "locked": "#6b7280",
    "ready": "#15803d",
    "idle": "#15803d",
    "dosing": "#1d4ed8",
    "active": "#1d4ed8",
    "running": "#1d4ed8",
    "paused": "#b45309",
    "completed": "#047857",
    "fault": "#b91c1c",
    "error": "#b91c1c",
}


class _ElidedLabel(QLabel):
    """Single-line label that elides instead of growing the layout."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = text
        self.setWordWrap(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(40)
        super().setText(text)

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._full = text
        self.setToolTip(text)
        self._elide()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        width = max(self.width(), 40)
        elided = self.fontMetrics().elidedText(
            self._full, Qt.TextElideMode.ElideMiddle, width
        )
        super().setText(elided)


class _Metric(QFrame):
    """Caption + value tile with a reserved height so live updates don't reflow."""

    def __init__(
        self,
        caption: str,
        parent: QWidget | None = None,
        *,
        tip: str = "",
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(68)
        if tip:
            self.setToolTip(tip)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        self._caption = QLabel(caption)
        caption_font = self._caption.font()
        caption_font.setPointSize(max(8, caption_font.pointSize() - 1))
        self._caption.setFont(caption_font)
        self._value = _ElidedLabel("—")
        value_font = QFont(self._value.font())
        value_font.setPointSize(max(12, value_font.pointSize() + 2))
        value_font.setBold(True)
        self._value.setFont(value_font)
        layout.addWidget(self._caption)
        layout.addWidget(self._value)

    def set_value(self, text: str) -> None:
        self._value.setText(text)


class _PercentBar(QWidget):
    """Rounded continuous bar — Windows QProgressBar chunks look dated."""

    def __init__(self, suffix: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._suffix = suffix
        self._pct: float | None = None
        self.setFixedHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_percent(self, value: float | None) -> None:
        self._pct = value
        if value is None:
            self.setToolTip(f"{self._suffix}: unknown")
        else:
            self.setToolTip(f"{self._suffix}: {value:.1f}%")
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0.5, 3.5, max(self.width() - 1.0, 1.0), max(self.height() - 7.0, 8.0))
        pal = self.palette()
        track = pal.color(pal.ColorRole.Midlight)
        fill_color = pal.color(pal.ColorRole.Highlight)
        clip = QPainterPath()
        clip.addRoundedRect(rect, 6.0, 6.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track)
        painter.drawPath(clip)
        if self._pct is not None and self._pct > 0:
            painter.save()
            painter.setClipPath(clip)
            fill = QRectF(
                rect.x(),
                rect.y(),
                max(rect.width() * min(self._pct, 100.0) / 100.0, 4.0),
                rect.height(),
            )
            painter.setBrush(fill_color)
            painter.drawRect(fill)
            painter.restore()
        if self._pct is None:
            label = f"—  {self._suffix}"
        else:
            label = f"{self._pct:.1f}%  {self._suffix}"
        painter.setPen(pal.color(pal.ColorRole.WindowText))
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), label)


class PlanRunnerPanel(QWidget):
    """Upload input_map.csv to RCI, run controls, and download session data."""

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
        self._status: dict = {}
        self._splitter_ready = False
        self._busy = False
        self._hold_footer = False
        self._session_remote_root = "/root/reports/session"
        self._build_ui()
        self._start_worker()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        root.addLayout(self._build_connection_bar())

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(6)
        self._splitter.addWidget(self._build_setup_pane())
        self._splitter.addWidget(self._build_monitor_pane())
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        root.addWidget(self._splitter, stretch=1)

        self._footer = _ElidedLabel("Enter an RCI host and connect.")
        self._footer.setMinimumHeight(self._footer.fontMetrics().height())
        root.addWidget(self._footer)
        self._apply_status({})
        self._refresh_upload_enabled()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._splitter_ready:
            return
        self._splitter_ready = True
        self._apply_splitter_sizes()

    def _apply_splitter_sizes(self) -> None:
        total = self._splitter.size().width()
        if total < _LEFT_PANE_MIN + _RIGHT_PANE_MIN:
            total = max(self.width() - 20, _LEFT_PANE_WIDTH + _RIGHT_PANE_MIN)
        left = min(_LEFT_PANE_WIDTH, max(_LEFT_PANE_MIN, total - _RIGHT_PANE_MIN))
        self._splitter.setSizes([left, max(total - left, _RIGHT_PANE_MIN)])

    def _build_connection_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(8)

        self._host_edit = QLineEdit()
        self._host_edit.setPlaceholderText("RCI IP — 192.168.100.184")
        self._host_edit.setToolTip(
            "IP of the RCI. A browser URL like http://192.168.100.184/io/ also works.\n"
            "Press Enter to connect."
        )
        self._host_edit.setClearButtonEnabled(True)
        ip_w = self._host_edit.fontMetrics().horizontalAdvance("192.168.100.184:80") + 36
        self._host_edit.setFixedWidth(max(200, ip_w))
        if self._app_settings.last_rci_host:
            self._host_edit.setText(self._app_settings.last_rci_host)
        self._host_edit.returnPressed.connect(self._on_connect_clicked)

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setDefault(True)
        disconnect_w = self._connect_btn.fontMetrics().horizontalAdvance("Disconnect") + 24
        self._connect_btn.setFixedWidth(max(110, disconnect_w))
        self._connect_btn.setToolTip("Connect to the RCI on port 80.")
        self._connect_btn.clicked.connect(self._on_connect_clicked)

        self._conn_dot = QLabel("●")
        self._conn_dot.setFixedWidth(16)
        self._conn_dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._conn_dot.setStyleSheet("color: #9ca3af; font-size: 16px;")
        self._conn_identity = QLabel("Disconnected")
        ident_font = QFont(self._conn_identity.font())
        ident_font.setBold(True)
        self._conn_identity.setFont(ident_font)
        ident_w = max(
            self._conn_identity.fontMetrics().horizontalAdvance("Disconnected"),
            self._conn_identity.fontMetrics().horizontalAdvance("Connecting…"),
        )
        self._conn_identity.setMinimumWidth(ident_w)
        self._conn_detail = _ElidedLabel("")

        status = QWidget()
        status.setMinimumWidth(260)
        status.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        status_l = QHBoxLayout(status)
        status_l.setContentsMargins(0, 0, 0, 0)
        status_l.setSpacing(6)
        status_l.addWidget(self._conn_dot)
        status_l.addWidget(self._conn_identity)
        status_l.addWidget(self._conn_detail, stretch=1)

        host_label = QLabel("Host")
        host_label.setToolTip(self._host_edit.toolTip())
        bar.addWidget(host_label)
        bar.addWidget(self._host_edit)
        bar.addWidget(self._connect_btn)
        bar.addSpacing(8)
        bar.addWidget(status, stretch=1)
        return bar

    def _hint_label(self, text: str) -> _ElidedLabel:
        label = _ElidedLabel(text)
        line = label.fontMetrics().height()
        label.setMinimumHeight(line)
        label.setMaximumHeight(line)
        return label

    def _build_setup_pane(self) -> QWidget:
        scroll = make_pane_scroll_area()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)

        plan_box = QGroupBox("Plan")
        plan_l = QVBoxLayout(plan_box)
        plan_row = QHBoxLayout()
        self._plan_path_edit = QLineEdit()
        self._plan_path_edit.setPlaceholderText("Plan Synthesis input_map.csv")
        self._plan_path_edit.setToolTip(
            "CSV from the Plan Synthesis tab. Browse or paste a path, then upload."
        )
        self._plan_path_edit.textChanged.connect(self._refresh_upload_enabled)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(
            browse_btn.fontMetrics().horizontalAdvance("Browse…") + 24
        )
        browse_btn.setToolTip("Pick an input_map.csv exported from Plan Synthesis.")
        browse_btn.clicked.connect(self._on_browse_plan)
        plan_row.addWidget(self._plan_path_edit, stretch=1)
        plan_row.addWidget(browse_btn)
        plan_l.addLayout(plan_row)
        self._upload_btn = QPushButton("Upload to RCI")
        self._upload_btn.setEnabled(False)
        self._upload_btn.clicked.connect(self._on_upload)
        plan_l.addWidget(self._upload_btn)
        self._plan_hint = self._hint_label("Connect, then choose a CSV to upload.")
        plan_l.addWidget(self._plan_hint)
        layout.addWidget(plan_box)

        run_box = QGroupBox("Run")
        run_l = QVBoxLayout(run_box)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._start_btn = QPushButton("Start")
        self._pause_btn = QPushButton("Pause")
        self._stop_btn = QPushButton("Stop")
        self._reset_btn = QPushButton("Reset")
        self._start_btn.setStyleSheet("font-weight: 600;")
        self._start_btn.setToolTip("Begin the uploaded plan on the RCI.")
        self._pause_btn.setToolTip("Pause after the current control point.")
        self._stop_btn.setToolTip("Stop the run. Use Reset before starting again.")
        self._reset_btn.setToolTip("Return the controller to idle so a new run can start.")
        for btn in (self._start_btn, self._pause_btn, self._stop_btn, self._reset_btn):
            btn.setEnabled(False)
            btn.setMinimumHeight(32)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn_row.addWidget(btn)
        self._start_btn.clicked.connect(lambda: self._invoke("start_run"))
        self._pause_btn.clicked.connect(lambda: self._invoke("pause_run"))
        self._stop_btn.clicked.connect(lambda: self._invoke("stop_run"))
        self._reset_btn.clicked.connect(lambda: self._invoke("reset_run"))
        run_l.addLayout(btn_row)
        self._permit_hint = self._hint_label("Connect first — Start enables when the RCI grants permit.")
        run_l.addWidget(self._permit_hint)
        layout.addWidget(run_box)

        session_box = QGroupBox("Session")
        session_l = QVBoxLayout(session_box)
        dest_row = QHBoxLayout()
        self._session_path_edit = QLineEdit()
        self._session_path_edit.setPlaceholderText("Save session zip as…")
        self._session_path_edit.setToolTip(
            "Local zip path. The file name is the session folder on the RCI "
            "(under /root/reports/session/). The zip is G3 layout so Data Analysis "
            "can open it like any other session."
        )
        dest_browse = QPushButton("Browse…")
        dest_browse.setFixedWidth(
            dest_browse.fontMetrics().horizontalAdvance("Browse…") + 24
        )
        dest_browse.setToolTip("Choose the folder and zip file name.")
        dest_browse.clicked.connect(self._on_browse_session_zip)
        self._session_path_edit.textChanged.connect(self._refresh_session_hint)
        dest_row.addWidget(self._session_path_edit, stretch=1)
        dest_row.addWidget(dest_browse)
        session_l.addLayout(dest_row)
        self._download_btn = QPushButton("Download")
        self._download_btn.setEnabled(False)
        self._download_btn.setToolTip(
            "Save the RCI session as a zip the Data Analysis tab can open."
        )
        self._download_btn.clicked.connect(self._on_download_session)
        session_l.addWidget(self._download_btn)
        self._session_hint = self._hint_label(
            "Zip name is the session folder on the RCI."
        )
        session_l.addWidget(self._session_hint)
        layout.addWidget(session_box)
        layout.addStretch(1)

        set_pane_scroll_widget(scroll, pane)
        scroll.setMinimumWidth(_LEFT_PANE_MIN)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        return scroll

    def _build_monitor_pane(self) -> QWidget:
        pane = QWidget()
        pane.setMinimumWidth(_RIGHT_PANE_MIN)
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(10)

        self._state_label = QLabel("—")
        state_font = QFont(self._state_label.font())
        state_font.setPointSize(max(22, state_font.pointSize() + 10))
        state_font.setBold(True)
        self._state_label.setFont(state_font)
        self._state_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self._state_label.setMinimumHeight(self._state_label.fontMetrics().height() + 4)
        self._state_sub = _ElidedLabel("Waiting for connection")
        self._live_stamp = QLabel("Updated —")
        self._live_stamp.setMinimumHeight(self._live_stamp.fontMetrics().height())
        layout.addWidget(self._state_label)
        layout.addWidget(self._state_sub)
        layout.addWidget(self._live_stamp)

        self._progress_bar = _PercentBar("overall")
        layout.addWidget(self._progress_bar)
        self._point_bar = _PercentBar("current point")
        layout.addWidget(self._point_bar)

        grid = QGridLayout()
        grid.setSpacing(8)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        self._metric_point = _Metric(
            "Control point",
            tip="Current spot in the uploaded plan.",
        )
        self._metric_energy = _Metric(
            "Energy",
            tip="Beam energy of the current control point.",
        )
        self._metric_layer = _Metric(
            "Layer",
            tip="Layer ID of the current control point.",
        )
        self._metric_elapsed = _Metric(
            "Elapsed",
            tip="Time the controller has been in this session.",
        )
        self._metric_permit = _Metric(
            "Start permit",
            tip="Controller ready permit — the same interlock the RCI page shows.",
        )
        self._metric_points_ok = _Metric(
            "Points",
            tip="Whether the controller accepted the uploaded plan.",
        )
        grid.addWidget(self._metric_point, 0, 0)
        grid.addWidget(self._metric_energy, 0, 1)
        grid.addWidget(self._metric_layer, 0, 2)
        grid.addWidget(self._metric_elapsed, 1, 0)
        grid.addWidget(self._metric_permit, 1, 1)
        grid.addWidget(self._metric_points_ok, 1, 2)
        layout.addLayout(grid)
        layout.addStretch(1)
        return pane

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
        worker.status_error.connect(self._on_status_error)
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
            QMetaObject.invokeMethod(worker, method, Qt.ConnectionType.QueuedConnection)
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

    def shutdown(self) -> None:
        worker = self._worker
        thread = self._worker_thread
        if worker is not None and thread is not None and thread.isRunning():
            QMetaObject.invokeMethod(
                worker, "disconnect_host", Qt.ConnectionType.QueuedConnection
            )
            thread.quit()
            thread.wait(3000)
        self._worker = None
        self._worker_thread = None

    def closeEvent(self, event: QCloseEvent) -> None:
        self.shutdown()
        super().closeEvent(event)

    def _set_footer(self, text: str, *, busy: bool | None = None) -> None:
        if busy is not None:
            self._busy = busy
            if busy:
                self._hold_footer = False
            self._refresh_action_enables()
        self._footer.setText(text)

    def _plan_path(self) -> str:
        return self._plan_path_edit.text().strip()

    def _refresh_upload_enabled(self) -> None:
        ready = self._connected and bool(self._plan_path()) and not self._busy
        self._upload_btn.setEnabled(ready)
        if not self._connected:
            self._upload_btn.setToolTip("Connect to the RCI first.")
        elif self._busy:
            self._upload_btn.setToolTip("Wait for the current operation to finish.")
        elif not self._plan_path():
            self._upload_btn.setToolTip("Choose an input_map.csv first.")
        else:
            self._upload_btn.setToolTip("Send this CSV to the RCI as control points.")

    def _refresh_download_enabled(self) -> None:
        self._download_btn.setEnabled(self._connected and not self._busy)
        if not self._connected:
            self._download_btn.setToolTip("Connect to the RCI first.")
        elif self._busy:
            self._download_btn.setToolTip("Wait for the current operation to finish.")
        else:
            self._download_btn.setToolTip(
                "Save the RCI session as a zip the Data Analysis tab can open."
            )

    def _refresh_action_enables(self) -> None:
        self._refresh_upload_enabled()
        self._refresh_download_enabled()
        self._set_run_controls(self._status)

    def _refresh_session_hint(self) -> None:
        self._session_hint.setText(
            session_download_hint(
                self._session_path_edit.text(), self._session_remote_root
            )
        )

    def _coach_footer(self, status: dict | None = None) -> None:
        if self._busy or self._hold_footer:
            return
        self._set_footer(self._next_step(status if status is not None else self._status))

    def _next_step(self, status: dict) -> str:
        return coach_message(
            connected=self._connected,
            has_plan=bool(self._plan_path()),
            status=status,
        )

    def _on_connect_clicked(self) -> None:
        if self._connected:
            self._connect_btn.setEnabled(False)
            self._invoke("disconnect_host")
            return

        raw = self._host_edit.text().strip()
        if not raw:
            QMessageBox.warning(self, "Plan Runner", "Enter an RCI host IP or URL.")
            return
        try:
            host = parse_host(raw)
        except ValueError as exc:
            QMessageBox.warning(self, "Plan Runner", str(exc))
            return
        if host != raw:
            self._host_edit.setText(host)
        self._hold_footer = False
        self._connect_btn.setEnabled(False)
        self._conn_dot.setStyleSheet("color: #d97706; font-size: 16px;")
        self._conn_identity.setText("Connecting…")
        self._set_footer(f"Connecting to {host}…", busy=True)
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
        self._hold_footer = False
        self._plan_path_edit.setText(path)
        self._remember_plan_dir(path)
        self._plan_hint.setText(Path(path).name)
        self._refresh_upload_enabled()
        self._coach_footer()

    def _remember_plan_dir(self, path: str | Path) -> None:
        try:
            self._app_settings.last_plan_runner_file_dir = str(Path(path).parent)
            self._app_settings.save()
        except OSError:
            pass

    def _on_upload(self) -> None:
        path = self._plan_path()
        if not path:
            QMessageBox.warning(self, "Plan Runner", "Choose an input_map.csv file.")
            return
        self._hold_footer = False
        self._upload_btn.setEnabled(False)
        self._set_footer(f"Uploading {Path(path).name}…", busy=True)
        self._invoke("upload_plan", path)

    def _default_session_zip_path(self) -> str:
        return default_session_zip_path(
            self._session_path_edit.text(),
            self._app_settings.last_plan_runner_file_dir or "",
        )

    def _on_browse_session_zip(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save session zip",
            self._default_session_zip_path(),
            "Zip archives (*.zip)",
        )
        if not path:
            return
        if not path.lower().endswith(".zip"):
            path += ".zip"
        self._session_path_edit.setText(path)
        self._remember_plan_dir(path)
        self._refresh_session_hint()

    def _on_download_session(self) -> None:
        dest = self._session_path_edit.text().strip()
        resolved = resolve_session_download(dest, self._session_remote_root)
        remote = dest.rstrip("/") if dest.startswith("/root/") else ""
        zip_path = ""
        if resolved is not None:
            remote, zip_path = resolved
        if not zip_path:
            self._on_browse_session_zip()
            dest = self._session_path_edit.text().strip()
            resolved = resolve_session_download(dest, self._session_remote_root)
            if resolved is None:
                return
            zip_path = resolved[1]
            if not remote:
                remote = resolved[0]
        self._hold_footer = False
        self._remember_plan_dir(zip_path)
        self._session_path_edit.setText(zip_path)
        self._set_footer("Downloading session…", busy=True)
        self._invoke("download_session", remote, zip_path)

    def _set_run_controls(self, status: dict) -> None:
        enables = control_enables(status, connected=self._connected and not self._busy)
        self._start_btn.setEnabled(enables["start"])
        self._pause_btn.setEnabled(enables["pause"])
        self._stop_btn.setEnabled(enables["stop"])
        self._reset_btn.setEnabled(enables["reset"])

        if not self._connected:
            self._start_btn.setToolTip("Connect to the RCI first.")
            self._permit_hint.setText("Connect first — Start enables when the RCI grants permit.")
            return
        if self._busy:
            self._start_btn.setToolTip("Wait for the current operation to finish.")
            return
        if enables["start"]:
            self._start_btn.setToolTip("Begin the uploaded plan on the RCI.")
            state = io_text(status.get(STATE)) or "unknown"
            self._permit_hint.setText(f"Start is ready · controller is {state}.")
            return
        if io_text(status.get(STATE)).lower() in {"dosing", "active", "running"}:
            self._start_btn.setToolTip("Already running — use Pause or Stop.")
            self._permit_hint.setText("Run in progress. Pause or Stop if you need to halt.")
            return
        if not io_bool(status.get(POINTS_VALID)):
            self._start_btn.setToolTip("Upload a valid plan first.")
            self._permit_hint.setText("Upload a plan — Start stays off until the RCI accepts it.")
            return
        reason = io_text(status.get(READY_PERMIT_REASON))
        if reason:
            self._start_btn.setToolTip(reason)
            self._permit_hint.setText(reason)
            return
        self._start_btn.setToolTip("The RCI ready permit is not granted yet.")
        self._permit_hint.setText("Plan is loaded. Waiting for the RCI ready permit.")

    def _on_connected(self, info: object) -> None:
        self._connected = True
        self._connect_btn.setText("Disconnect")
        self._connect_btn.setEnabled(True)
        self._connect_btn.setToolTip("Disconnect from this RCI.")
        self._host_edit.setEnabled(False)
        version = info.get("version") if isinstance(info, dict) else ""
        device = info.get("device_type") if isinstance(info, dict) else ""
        host = info.get("host") if isinstance(info, dict) else ""
        self._conn_dot.setStyleSheet("color: #16a34a; font-size: 16px;")
        self._conn_identity.setText(str(device or "IGX"))
        self._conn_detail.setText(f"{host}  ·  {version}".strip(" ·"))
        self._busy = False
        self._hold_footer = False
        self._refresh_action_enables()
        if isinstance(info, dict):
            session_dir = info.get("session_directory")
            if session_dir:
                self._session_remote_root = str(session_dir).rstrip("/")
            self._refresh_session_hint()
            status = info.get("status")
            if isinstance(status, dict):
                self._on_status_updated(status)
                return
        self._coach_footer(info.get("status") if isinstance(info, dict) else {})

    def _on_connection_failed(self, message: str) -> None:
        self._reset_connection_chrome()
        self._set_footer(f"Connection failed: {message}", busy=False)
        QMessageBox.critical(self, "Connection failed", message)

    def _on_disconnected(self) -> None:
        self._reset_connection_chrome()
        self._busy = False
        self._coach_footer({})
        self._apply_status({})

    def _reset_connection_chrome(self) -> None:
        self._connected = False
        self._status = {}
        self._connect_btn.setText("Connect")
        self._connect_btn.setEnabled(True)
        self._connect_btn.setToolTip("Connect to the RCI on port 80.")
        self._host_edit.setEnabled(True)
        self._conn_dot.setStyleSheet("color: #9ca3af; font-size: 16px;")
        self._conn_identity.setText("Disconnected")
        self._conn_detail.setText("")
        self._hold_footer = False
        self._refresh_action_enables()

    def _on_upload_done(self, target: str) -> None:
        self._set_footer(f"Plan uploaded to {target}.", busy=False)
        name = Path(self._plan_path()).name or "plan"
        self._plan_hint.setText(f"{name} loaded on the RCI")
        self._coach_footer()

    def _on_upload_failed(self, message: str) -> None:
        self._set_footer(f"Upload failed: {message}", busy=False)
        QMessageBox.critical(self, "Upload failed", message)

    def _on_operation_failed(self, message: str) -> None:
        self._set_footer(message, busy=False)
        QMessageBox.warning(self, "Plan Runner", message)

    def _on_status_error(self, message: str) -> None:
        if not self._busy:
            self._set_footer(f"Status poll failed: {message}")

    def _on_status_updated(self, status: object) -> None:
        if not isinstance(status, dict):
            return
        self._status = status
        self._apply_status(status)

    def _apply_status(self, status: dict) -> None:
        state = io_text(status.get(STATE))
        state_text = state.upper() if state else "—"
        self._state_label.setText(state_text)
        color = _STATE_COLORS.get(state.lower(), "#374151")
        self._state_label.setStyleSheet(f"color: {color};")

        combined = unwrap_io(status.get(COMBINED_STATE))
        parts = []
        if combined not in (None, "", False):
            parts.append(f"Map manager: {combined}")
        if io_bool(status.get(TREATMENT_ACTIVE)):
            parts.append("Treatment active")
        if not self._connected:
            self._state_sub.setText("Connect, upload a plan, then press Start")
        elif parts:
            self._state_sub.setText("  ·  ".join(parts))
        else:
            self._state_sub.setText("Waiting for live status")
        if status:
            self._live_stamp.setText(datetime.now().strftime("Updated %H:%M:%S"))
        else:
            self._live_stamp.setText("Updated —")

        self._progress_bar.set_percent(progress_percent(status.get(PROGRESS)))
        self._point_bar.set_percent(progress_percent(status.get(POINT_PROGRESS)))

        self._metric_point.set_value(point_fraction(status))
        self._metric_energy.set_value(format_energy(status.get(POINT_ENERGY)))
        layer = unwrap_io(status.get(POINT_LAYER_ID))
        self._metric_layer.set_value(str(layer) if layer not in (None, "") else "—")
        self._metric_elapsed.set_value(format_elapsed(status.get(TIME_ELAPSED)))
        if not status:
            self._metric_permit.set_value("—")
        else:
            self._metric_permit.set_value(
                "Granted" if io_bool(status.get(READY_PERMIT)) else "Held"
            )
        if io_bool(status.get(POINTS_VALID)) or io_bool(status.get(COMBINED_POINTS_OK)):
            self._metric_points_ok.set_value("Valid")
        elif status:
            self._metric_points_ok.set_value("Invalid")
        else:
            self._metric_points_ok.set_value("—")

        count = io_number(status.get(CONTROL_POINT_COUNT))
        local = Path(self._plan_path()).name
        if count and self._connected:
            shown = int(count) if count.is_integer() else count
            prefix = f"{local} · " if local else ""
            self._plan_hint.setText(f"{prefix}{shown} control points on the RCI")

        self._set_run_controls(status)
        self._refresh_download_enabled()
        self._coach_footer(status)

    def _on_download_done(self, path: str) -> None:
        self._hold_footer = True
        self._set_footer(f"Session saved to {path}", busy=False)

    def _on_download_failed(self, message: str) -> None:
        self._set_footer(f"Download failed: {message}", busy=False)
        QMessageBox.critical(self, "Download failed", message)
