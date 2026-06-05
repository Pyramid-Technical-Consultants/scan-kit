"""Auto-tuning workflow list (left) and detail form (right)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from scan_kit.common.session_browser import SessionBrowserWidget, default_project_root

from .auto_tuning.base import AutoTuneRunResult, AutoTuneWorkflow
from .auto_tuning.params import AutoTuneParamSpec
from .auto_tuning.registry import AUTO_TUNE_REGISTRY

ApplyFn = Callable[[AutoTuneWorkflow, dict[str, Any]], AutoTuneRunResult | None]


class _WorkflowListRow(QWidget):
    """One workflow list entry: name on the left, muted description on the right."""

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


class AutoTuneListWidget(QWidget):
    """Workflow picker shown in the configuration tuning left pane."""

    workflow_selected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._workflows = list(AUTO_TUNE_REGISTRY)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        heading = QLabel("Auto tuning")
        heading.setStyleSheet("font-weight: 600;")
        layout.addWidget(heading)

        self._workflow_list = QListWidget()
        self._workflow_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        for workflow in self._workflows:
            row = _WorkflowListRow(workflow.name, workflow.description)
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, workflow.id)
            item.setSizeHint(row.sizeHint())
            self._workflow_list.addItem(item)
            self._workflow_list.setItemWidget(item, row)
        self._workflow_list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._workflow_list)

    def select_first(self) -> None:
        if self._workflows:
            self._workflow_list.setCurrentRow(0)

    def clear_selection(self) -> None:
        self._workflow_list.blockSignals(True)
        self._workflow_list.setCurrentRow(-1)
        self._workflow_list.blockSignals(False)

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._workflows):
            self.workflow_selected.emit(None)
            return
        self.workflow_selected.emit(self._workflows[row])


class AutoTuneDetailWidget(QWidget):
    """Workflow parameters and actions shown in the right pane."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current: AutoTuneWorkflow | None = None
        self._editors: dict[str, QLineEdit] = {}
        self._default_data_dir = ""
        self._apply_fn: ApplyFn | None = None
        self._session_browser: SessionBrowserWidget | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._description_label = QLabel()
        self._description_label.setWordWrap(True)
        self._description_label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        layout.addWidget(self._description_label)

        action_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply to devices.xml")
        self._apply_btn.setToolTip(
            "Rewrite beam_sigma K0 values in the open configuration's devices.xml "
            "from the selected session (marks the file dirty until you save)."
        )
        self._apply_btn.clicked.connect(self._on_apply)
        action_row.addWidget(self._apply_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self._param_host = QWidget()
        self._param_form = QFormLayout(self._param_host)
        self._param_form.setContentsMargins(0, 8, 0, 0)
        self._param_form.setSpacing(6)
        self._param_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        layout.addWidget(self._param_host, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addStretch(1)

    def shutdown(self) -> None:
        if self._session_browser is not None:
            self._session_browser.shutdown()

    def set_apply_handler(self, handler: ApplyFn | None) -> None:
        self._apply_fn = handler

    def set_default_data_dir(self, path: str) -> None:
        self._default_data_dir = path.strip()
        if "data_dir" in self._editors and not self._editors["data_dir"].text().strip():
            self._editors["data_dir"].setText(self._default_data_dir)
        if self._session_browser is not None and self._default_data_dir:
            self._session_browser.set_base_dir(self._default_data_dir, refresh=False)

    def set_workflow(self, workflow: AutoTuneWorkflow | None) -> None:
        self._current = workflow
        if workflow is None:
            self._description_label.clear()
            self._clear_param_form()
            self._hide_session_browser()
            self._apply_btn.setEnabled(False)
            return
        self._description_label.setText(workflow.description)
        self._apply_btn.setEnabled(True)
        if workflow.uses_session_browser():
            self._show_session_browser()
        else:
            self._hide_session_browser()
            self._rebuild_param_form()

    def _ensure_session_browser(self) -> SessionBrowserWidget:
        if self._session_browser is None:
            initial = self._default_data_dir or str(default_project_root() / "test_data")
            self._session_browser = SessionBrowserWidget(
                project_root=default_project_root(),
                initial_base_dir=initial,
                max_selections=1,
                editable_notes=False,
                show_plot_swatches=False,
                parent=self,
            )
            self.layout().insertWidget(2, self._session_browser, stretch=1)
            if self._default_data_dir:
                self._session_browser.refresh()
        return self._session_browser

    def _show_session_browser(self) -> None:
        browser = self._ensure_session_browser()
        browser.setVisible(True)
        self._param_host.setVisible(False)
        if self._default_data_dir:
            browser.set_base_dir(self._default_data_dir, refresh=True)

    def _hide_session_browser(self) -> None:
        if self._session_browser is not None:
            self._session_browser.setVisible(False)
        self._param_host.setVisible(True)

    def _clear_param_form(self) -> None:
        while self._param_form.rowCount():
            self._param_form.removeRow(0)
        self._editors.clear()

    def _rebuild_param_form(self) -> None:
        self._clear_param_form()
        if self._current is None:
            return

        defaults = self._current.default_params()
        for spec in self._current.param_specs():
            field = self._make_editor(spec, defaults.get(spec.key, spec.default))
            self._param_form.addRow(spec.label, field)

    def _make_editor(self, spec: AutoTuneParamSpec, initial: str) -> QWidget:
        if spec.kind == "directory":
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            line = QLineEdit()
            value = str(initial).strip() or self._default_data_dir
            line.setText(value)
            if spec.placeholder:
                line.setPlaceholderText(spec.placeholder)
            browse = QPushButton("…")
            browse.setFixedWidth(28)
            browse.setToolTip("Browse for session data folder")
            browse.clicked.connect(lambda: self._browse_directory(line))
            row_layout.addWidget(line, stretch=1)
            row_layout.addWidget(browse)
            self._register_line_editor(spec.key, line)
            return row

        line = QLineEdit()
        line.setText(str(initial))
        if spec.placeholder:
            line.setPlaceholderText(spec.placeholder)
        self._register_line_editor(spec.key, line)
        return line

    def _register_line_editor(self, key: str, line: QLineEdit) -> None:
        self._editors[key] = line

    def _browse_directory(self, target: QLineEdit) -> None:
        start = target.text().strip() or self._default_data_dir or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select session data folder",
            start,
        )
        if chosen:
            target.setText(chosen)

    def read_params(self) -> dict[str, Any]:
        params = {key: editor.text().strip() for key, editor in self._editors.items()}
        if (
            self._current is not None
            and self._current.uses_session_browser()
            and self._session_browser is not None
        ):
            selected = self._session_browser.selected_session_ids()
            params["session_id"] = selected[0] if selected else ""
            params["data_dir"] = self._session_browser.base_dir()
        return params

    def _on_apply(self) -> None:
        if self._current is None:
            return
        params = self.read_params()
        errors = self._current.validate(params)
        if errors:
            QMessageBox.warning(self, "Auto tuning", "\n".join(errors))
            return
        if self._apply_fn is None:
            QMessageBox.warning(
                self,
                "Auto tuning",
                "Configuration folder is not open.",
            )
            return
        result = self._apply_fn(self._current, params)
        if result is None:
            return
        if result.warnings:
            detail = result.message + "\n\n" + "\n".join(result.warnings)
        else:
            detail = result.message
        if result.success:
            QMessageBox.information(self, "Auto tuning", detail)
        else:
            QMessageBox.warning(self, "Auto tuning", detail)
