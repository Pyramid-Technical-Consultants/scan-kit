"""Plan Synthesis workflow panel."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QVBoxLayout,
    QWidget,
    QLabel,
)

import pandas as pd

from ..common.app_settings import AppSettings
from ..common.qt_widgets import (
    configure_pane_scroll_area,
    make_pane_scroll_area,
    set_pane_scroll_widget,
)
from .plan_synthesis.base import PlanTemplate
from .plan_synthesis.export_filename import suggest_input_map_filename
from .plan_synthesis.generation_worker import PlanGenerationWorker
from .plan_synthesis.input_map import write_input_map_csv
from .plan_synthesis.param_form import ParamFormWidget
from .plan_synthesis.paths import resolve_plan_synthesis_save_dir
from .plan_synthesis.preview import (
    clear_preview_table,
    fill_preview_table,
    format_plan_summary,
    make_summary_label,
    start_preview_table_fill,
)
from .plan_synthesis.registry import TEMPLATE_REGISTRY


class _TemplateListRow(QWidget):
    """One template list entry: name on the left, muted description on the right."""

    def __init__(
        self,
        name: str,
        description: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(12)

        self._description = description

        name_label = QLabel(name)
        name_label.setSizePolicy(
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Preferred,
        )

        self._desc_label = QLabel(description)
        self._desc_label.setWordWrap(False)
        self._desc_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._desc_label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        self._desc_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self._desc_label.setToolTip(description)

        layout.addWidget(
            name_label,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        layout.addWidget(
            self._desc_label,
            stretch=1,
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        self.setMinimumHeight(32)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._update_description_elide()

    def _update_description_elide(self) -> None:
        width = self._desc_label.width()
        if width <= 0:
            return
        elided = self._desc_label.fontMetrics().elidedText(
            self._description,
            Qt.TextElideMode.ElideRight,
            width,
        )
        self._desc_label.setText(elided)


class PlanSynthesisPanel(QWidget):
    """Plan synthesis UI: templates + parameters (left) | preview (right)."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        app_settings: AppSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self._app_settings = (
            app_settings if app_settings is not None else AppSettings.load()
        )
        self._templates = list(TEMPLATE_REGISTRY)
        self._current: PlanTemplate | None = None
        self._param_form: ParamFormWidget | None = None
        self._generated: pd.DataFrame | None = None
        self._generate_thread: QThread | None = None
        self._generate_worker: PlanGenerationWorker | None = None
        self._progress_dialog: QProgressDialog | None = None
        self._generating = False
        self._preview_fill_token = 0
        self._last_generate_params: dict | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        outer_splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(outer_splitter)

        # --- Left: template list | parameters (vertical splitter) ---
        left_splitter = QSplitter(Qt.Orientation.Vertical)

        template_host = QWidget()
        template_l = QVBoxLayout(template_host)
        template_l.setContentsMargins(4, 4, 4, 4)
        self._template_list = QListWidget()
        self._template_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        for template in self._templates:
            row = _TemplateListRow(template.name, template.description)
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, template.id)
            item.setSizeHint(row.sizeHint())
            self._template_list.addItem(item)
            self._template_list.setItemWidget(item, row)
        self._template_list.currentRowChanged.connect(self._on_template_changed)
        template_l.addWidget(self._template_list)
        left_splitter.addWidget(template_host)

        self._params_host = QWidget()
        params_host = self._params_host
        params_l = QVBoxLayout(params_host)
        params_l.setContentsMargins(4, 4, 4, 4)
        params_l.setSpacing(6)

        self._param_scroll = make_pane_scroll_area()
        self._param_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._param_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._param_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        params_l.addWidget(self._param_scroll, stretch=1)

        left_splitter.addWidget(params_host)
        configure_pane_scroll_area(self._param_scroll, host=self._params_host)

        left_splitter.setStretchFactor(0, 0)
        left_splitter.setStretchFactor(1, 1)
        left_splitter.setSizes([120, 520])

        # --- Right: summary, actions, preview table ---
        preview_host = QWidget()
        preview_l = QVBoxLayout(preview_host)
        preview_l.setContentsMargins(4, 4, 4, 4)

        self._summary_label = make_summary_label()

        action_row = QHBoxLayout()
        self._generate_btn = QPushButton("Generate")
        self._generate_btn.clicked.connect(self._on_generate)
        self._save_btn = QPushButton("Save CSV…")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        action_row.addWidget(self._generate_btn)
        action_row.addWidget(self._save_btn)
        action_row.addStretch(1)
        preview_l.addLayout(action_row)

        self._preview_table = QTableWidget()
        self._preview_table.setAlternatingRowColors(True)
        self._preview_table.verticalHeader().setVisible(False)
        fill_preview_table(self._preview_table, None)
        preview_l.addWidget(self._preview_table, stretch=1)
        preview_l.addWidget(self._summary_label)

        outer_splitter.addWidget(left_splitter)
        outer_splitter.addWidget(preview_host)
        outer_splitter.setStretchFactor(0, 0)
        outer_splitter.setStretchFactor(1, 1)
        outer_splitter.setSizes([640, 760])

        if self._templates:
            self._template_list.setCurrentRow(0)

    def _on_template_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._templates):
            self._current = None
            return
        self._current = self._templates[row]
        self._generated = None
        self._last_generate_params = None
        self._save_btn.setEnabled(False)
        self._summary_label.setText("No plan generated yet.")
        self._preview_fill_token += 1
        clear_preview_table(self._preview_table)
        self._rebuild_param_form()

    def _rebuild_param_form(self) -> None:
        old_widget = self._param_scroll.takeWidget()
        if old_widget is not None:
            old_widget.deleteLater()
        self._param_form = None
        if self._current is None:
            return
        form = ParamFormWidget(
            self._current.param_specs(),
            self._current.default_params(),
        )
        self._param_form = form
        set_pane_scroll_widget(
            self._param_scroll,
            form,
            host=self._params_host,
        )

    def _read_params(self) -> dict:
        if self._param_form is None or self._current is None:
            return {}
        return self._param_form.read_params()

    def _on_generate(self) -> None:
        if self._current is None or self._generating:
            return
        params = self._read_params()
        errors = self._current.validate(params)
        if errors:
            QMessageBox.warning(self, "Invalid parameters", "\n".join(errors))
            return

        self._generating = True
        self._preview_fill_token += 1
        clear_preview_table(self._preview_table)
        self._last_generate_params = dict(params)
        self._generate_btn.setEnabled(False)
        self._save_btn.setEnabled(False)

        progress = QProgressDialog("Generating plan…", None, 0, 100, self)
        progress.setWindowTitle("Plan Synthesis")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setCancelButton(None)
        progress.setValue(0)
        self._progress_dialog = progress

        thread = QThread(self)
        worker = PlanGenerationWorker(self._current, params)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(progress.setValue)
        worker.finished.connect(self._on_generate_finished)
        worker.failed.connect(self._on_generate_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(progress.close)
        worker.failed.connect(progress.close)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_generate_thread)
        self._generate_thread = thread
        self._generate_worker = worker
        thread.start()
        progress.show()

    def _close_progress_dialog(self) -> None:
        if self._progress_dialog is None:
            return
        self._progress_dialog.setValue(100)
        self._progress_dialog.close()
        self._progress_dialog = None

    def _release_generation_state(self) -> None:
        """Allow another Generate run once background work has finished."""
        self._generating = False
        self._generate_btn.setEnabled(True)

    def _on_generate_finished(self, df: pd.DataFrame) -> None:
        self._close_progress_dialog()
        self._release_generation_state()
        self._generated = df
        self._save_btn.setEnabled(not df.empty)
        self._summary_label.setText(format_plan_summary(df))
        self._preview_fill_token += 1
        token = self._preview_fill_token
        start_preview_table_fill(
            self._preview_table,
            df,
            is_current=lambda: self._preview_fill_token == token,
        )

    def _on_generate_failed(self, message: str) -> None:
        self._close_progress_dialog()
        self._release_generation_state()
        QMessageBox.critical(self, "Generation failed", message)

    def _clear_generate_thread(self) -> None:
        self._generate_thread = None
        self._generate_worker = None

    def _default_save_filename(self) -> str:
        if self._current is None:
            return "input_map.csv"
        params = self._last_generate_params or self._read_params()
        return suggest_input_map_filename(self._current, params)

    def _default_save_path(self) -> str:
        directory = resolve_plan_synthesis_save_dir(
            self._app_settings.last_plan_synthesis_save_dir
        )
        return str(directory / self._default_save_filename())

    def _remember_save_dir(self, path: str | Path) -> None:
        try:
            self._app_settings.last_plan_synthesis_save_dir = str(Path(path).parent)
            self._app_settings.save()
        except Exception:
            pass

    def _on_save(self) -> None:
        if self._generated is None or self._generated.empty:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save input map CSV",
            self._default_save_path(),
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            write_input_map_csv(self._generated, path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._remember_save_dir(path)
        QMessageBox.information(self, "Saved", f"Wrote {path}")
