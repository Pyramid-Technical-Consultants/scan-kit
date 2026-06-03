"""Plan Synthesis workflow panel."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

import pandas as pd

from .plan_synthesis.base import PlanTemplate
from .plan_synthesis.input_map import write_input_map_csv
from .plan_synthesis.param_form import ParamFormWidget
from .plan_synthesis.preview import fill_preview_table, format_plan_summary, make_summary_label
from .plan_synthesis.registry import TEMPLATE_REGISTRY


class PlanSynthesisPanel(QWidget):
    """Plan synthesis UI: templates + parameters (left) | preview (right)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._templates = list(TEMPLATE_REGISTRY)
        self._current: PlanTemplate | None = None
        self._param_form: ParamFormWidget | None = None
        self._generated: pd.DataFrame | None = None

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
            item = QListWidgetItem(template.name)
            item.setData(Qt.ItemDataRole.UserRole, template.id)
            self._template_list.addItem(item)
        self._template_list.currentRowChanged.connect(self._on_template_changed)
        template_l.addWidget(self._template_list)
        left_splitter.addWidget(template_host)

        params_host = QWidget()
        params_l = QVBoxLayout(params_host)
        params_l.setContentsMargins(4, 4, 4, 4)

        self._desc_label = QLabel("")
        self._desc_label.setWordWrap(True)
        params_l.addWidget(self._desc_label)

        self._param_host = QWidget()
        self._param_host_l = QVBoxLayout(self._param_host)
        self._param_host_l.setContentsMargins(0, 0, 0, 0)
        params_l.addWidget(self._param_host)
        params_l.addStretch(1)

        left_splitter.addWidget(params_host)

        left_splitter.setStretchFactor(0, 0)
        left_splitter.setStretchFactor(1, 1)
        left_splitter.setSizes([140, 520])

        # --- Right: summary, actions, preview table ---
        preview_host = QWidget()
        preview_l = QVBoxLayout(preview_host)
        preview_l.setContentsMargins(4, 4, 4, 4)

        self._summary_label = make_summary_label()
        preview_l.addWidget(self._summary_label)

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
        self._save_btn.setEnabled(False)
        self._summary_label.setText("No plan generated yet.")
        fill_preview_table(self._preview_table, None)
        self._desc_label.setText(self._current.description)
        self._rebuild_param_form()

    def _rebuild_param_form(self) -> None:
        while self._param_host_l.count():
            item = self._param_host_l.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._param_form = None
        if self._current is None:
            return
        form = ParamFormWidget(
            self._current.param_specs(),
            self._current.default_params(),
        )
        self._param_form = form
        self._param_host_l.addWidget(form)

    def _read_params(self) -> dict:
        if self._param_form is None or self._current is None:
            return {}
        return self._param_form.read_params()

    def _on_generate(self) -> None:
        if self._current is None:
            return
        params = self._read_params()
        errors = self._current.validate(params)
        if errors:
            QMessageBox.warning(self, "Invalid parameters", "\n".join(errors))
            return
        try:
            df = self._current.generate(params)
        except Exception as exc:
            QMessageBox.critical(self, "Generation failed", str(exc))
            return
        self._generated = df
        self._save_btn.setEnabled(True)
        self._summary_label.setText(format_plan_summary(df))
        fill_preview_table(self._preview_table, df)

    def _on_save(self) -> None:
        if self._generated is None or self._generated.empty:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save input map CSV",
            "input_map.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            write_input_map_csv(self._generated, path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Wrote {path}")
