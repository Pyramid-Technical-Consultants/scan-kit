"""Preview helpers for generated input_map plans."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView, QLabel, QTableWidget, QTableWidgetItem

from .input_map import plan_summary

_PREVIEW_COLUMNS = ("ENERGY", "X_POSITION", "Y_POSITION", "CHARGE_REQ", "spot_no")
PREVIEW_ROW_CAP = 1_000_000


def format_plan_summary(
    df: pd.DataFrame,
    *,
    preview_row_cap: int = PREVIEW_ROW_CAP,
) -> str:
    n_layers, n_spots, total_mu = plan_summary(df)
    if n_spots == 0:
        return "No plan generated yet."
    layer_word = "layer" if n_layers == 1 else "layers"
    spot_word = "spot" if n_spots == 1 else "spots"
    msg = f"{n_layers} {layer_word} · {n_spots} {spot_word} · {total_mu:.4f} MU total"
    if n_spots > preview_row_cap:
        msg += f" · preview shows first {preview_row_cap:,} rows"
    return msg


def _format_cell(col: str, val: object) -> str:
    if col == "CHARGE_REQ":
        return f"{float(val):.4f}"
    if col in ("ENERGY", "X_POSITION", "Y_POSITION"):
        return f"{float(val):g}"
    return str(int(val))  # spot_no


def fill_preview_table(table: QTableWidget, df: pd.DataFrame | None) -> int:
    """Populate *table* with all rows of *df*, up to :data:`PREVIEW_ROW_CAP`.

    Returns the number of rows displayed.
    """
    if df is None or df.empty:
        table.clear()
        table.setRowCount(0)
        table.setColumnCount(len(_PREVIEW_COLUMNS))
        table.setHorizontalHeaderLabels(list(_PREVIEW_COLUMNS))
        return 0

    n_show = min(len(df), PREVIEW_ROW_CAP)
    preview = df.iloc[:n_show]

    table.setSortingEnabled(False)
    table.setColumnCount(len(_PREVIEW_COLUMNS))
    table.setHorizontalHeaderLabels(list(_PREVIEW_COLUMNS))
    table.setRowCount(n_show)

    flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
    table.setUpdatesEnabled(False)
    try:
        for col_idx, col in enumerate(_PREVIEW_COLUMNS):
            column = preview[col].values
            for row_idx in range(n_show):
                item = QTableWidgetItem(_format_cell(col, column[row_idx]))
                item.setFlags(flags)
                table.setItem(row_idx, col_idx, item)
    finally:
        table.setUpdatesEnabled(True)

    hh = table.horizontalHeader()
    for c in range(len(_PREVIEW_COLUMNS)):
        hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
    table.setSortingEnabled(False)
    return n_show


def make_summary_label() -> QLabel:
    label = QLabel("No plan generated yet.")
    label.setWordWrap(True)
    return label
