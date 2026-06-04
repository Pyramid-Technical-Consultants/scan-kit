"""Shared energy-layer multiselect widget for plan synthesis templates."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .energies import (
    STANDARD_ENERGIES_MEV,
    TEN_MEV_STEP_ENERGIES_MEV,
    WHOLE_MEV_STEP_ENERGIES_MEV,
)
from .params import normalize_selected_energies

ENERGY_LIST_VISIBLE_ROWS = 8


def _list_height_for_rows(list_widget: QListWidget, rows: int) -> int:
    """Pixel height to show ``rows`` list entries (scroll for the rest)."""
    if list_widget.count() == 0:
        return 0
    row_h = list_widget.sizeHintForRow(0)
    if row_h <= 0:
        row_h = list_widget.fontMetrics().height() + 4
    return row_h * rows + 2 * list_widget.frameWidth()


def _populate_energy_list(
    energy_list: QListWidget,
    *,
    selected: set[float],
) -> None:
    energy_list.clear()
    for energy in reversed(STANDARD_ENERGIES_MEV):
        item = QListWidgetItem(f"{energy:g} MeV")
        item.setData(Qt.ItemDataRole.UserRole, float(energy))
        energy_list.addItem(item)
        if float(energy) in selected:
            item.setSelected(True)


def _select_energies(energy_list: QListWidget, targets: set[float]) -> None:
    energy_list.clearSelection()
    for index in range(energy_list.count()):
        item = energy_list.item(index)
        if item is None:
            continue
        value = item.data(Qt.ItemDataRole.UserRole)
        if value is not None and float(value) in targets:
            item.setSelected(True)


def _format_layer_selection_status(n_selected: int) -> str:
    if n_selected == 0:
        return "No layers selected"
    if n_selected == 1:
        return "1 layer selected"
    return f"{n_selected} layers selected"


class EnergyPickerWidget(QWidget):
    """Multiselect list of standard beam energies with preset selection buttons."""

    def __init__(
        self,
        selected: list[float] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        selected_set = {float(e) for e in selected} if selected else set()

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        select_all = QPushButton("Select All")
        select_whole_mev = QPushButton("Whole MeV Steps")
        select_ten_mev = QPushButton("10 MeV Steps")
        clear_all = QPushButton("Clear All")
        btn_row.addWidget(select_all)
        btn_row.addWidget(select_whole_mev)
        btn_row.addWidget(select_ten_mev)
        btn_row.addWidget(clear_all)

        self._status_label = QLabel()
        self._status_label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        self._status_label.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        btn_row.addWidget(self._status_label)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        self._energy_list = QListWidget()
        self._energy_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        _populate_energy_list(self._energy_list, selected=selected_set)
        self._energy_list.setFixedHeight(
            _list_height_for_rows(self._energy_list, ENERGY_LIST_VISIBLE_ROWS)
        )
        self._energy_list.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        lay.addWidget(self._energy_list)
        self.setFixedHeight(self.sizeHint().height())

        select_all.clicked.connect(self._energy_list.selectAll)
        select_whole_mev.clicked.connect(self._select_whole_mev_steps)
        select_ten_mev.clicked.connect(self._select_ten_mev_steps)
        clear_all.clicked.connect(self._energy_list.clearSelection)
        self._energy_list.itemSelectionChanged.connect(self._update_status_label)
        self._update_status_label()

    def _update_status_label(self) -> None:
        self._status_label.setText(
            _format_layer_selection_status(len(self.selected_energies()))
        )

    def _select_energy_preset(self, energies: tuple[float, ...]) -> None:
        _select_energies(
            self._energy_list,
            {float(energy) for energy in energies},
        )
        self._update_status_label()

    def _select_whole_mev_steps(self) -> None:
        self._select_energy_preset(WHOLE_MEV_STEP_ENERGIES_MEV)

    def _select_ten_mev_steps(self) -> None:
        self._select_energy_preset(TEN_MEV_STEP_ENERGIES_MEV)

    def selected_energies(self) -> list[float]:
        selected: list[float] = []
        for index in range(self._energy_list.count()):
            item = self._energy_list.item(index)
            if item is not None and item.isSelected():
                value = item.data(Qt.ItemDataRole.UserRole)
                if value is not None:
                    selected.append(float(value))
        return normalize_selected_energies(
            selected,
            catalog=STANDARD_ENERGIES_MEV,
        )
