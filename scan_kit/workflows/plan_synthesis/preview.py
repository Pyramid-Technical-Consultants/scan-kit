"""Preview helpers for generated input_map plans."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHeaderView, QLabel, QTableWidget, QTableWidgetItem

from .input_map import plan_summary

_PREVIEW_COLUMNS = ("ENERGY", "X_POSITION", "Y_POSITION", "CHARGE_REQ", "spot_no")
PREVIEW_ROW_CAP = 5_000
_PREVIEW_BATCH_SIZE = 200
_RESIZE_TO_CONTENTS_MAX_ROWS = 1_000


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


def clear_preview_table(table: QTableWidget) -> None:
    """Drop all preview rows and release table item memory."""
    table.setSortingEnabled(False)
    table.clearContents()
    table.setRowCount(0)
    table.setColumnCount(len(_PREVIEW_COLUMNS))
    table.setHorizontalHeaderLabels(list(_PREVIEW_COLUMNS))


def _apply_preview_header_modes(table: QTableWidget, *, n_rows: int) -> None:
    hh = table.horizontalHeader()
    if n_rows <= _RESIZE_TO_CONTENTS_MAX_ROWS:
        for col in range(len(_PREVIEW_COLUMNS)):
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        return
    for col in range(len(_PREVIEW_COLUMNS)):
        hh.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
    hh.setStretchLastSection(True)


def fill_preview_table(table: QTableWidget, df: pd.DataFrame | None) -> int:
    """Synchronously populate *table* (small/empty previews only)."""
    if df is None or df.empty:
        clear_preview_table(table)
        return 0

    n_show = min(len(df), PREVIEW_ROW_CAP)
    preview = df.iloc[:n_show]

    table.setSortingEnabled(False)
    table.clearContents()
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

    _apply_preview_header_modes(table, n_rows=n_show)
    table.setSortingEnabled(False)
    return n_show


def _count_filled_preview_cells(table: QTableWidget) -> int:
    """Return how many preview cells currently hold items."""
    n_cols = table.columnCount()
    if n_cols == 0:
        return 0
    filled = 0
    for row_idx in range(table.rowCount()):
        for col_idx in range(n_cols):
            if table.item(row_idx, col_idx) is not None:
                filled += 1
    return filled


def start_preview_table_fill(
    table: QTableWidget,
    df: pd.DataFrame | None,
    *,
    is_current: Callable[[], bool],
) -> None:
    """Fill the preview table in batches so the UI stays responsive."""
    if df is None or df.empty:
        clear_preview_table(table)
        return

    n_show = min(len(df), PREVIEW_ROW_CAP)
    preview = df.iloc[:n_show]
    columns = {col: preview[col].values for col in _PREVIEW_COLUMNS}

    table.setSortingEnabled(False)
    table.clearContents()
    table.setColumnCount(len(_PREVIEW_COLUMNS))
    table.setHorizontalHeaderLabels(list(_PREVIEW_COLUMNS))
    table.setRowCount(n_show)
    table.setUpdatesEnabled(False)

    flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
    state: dict[str, Any] = {"next_row": 0}

    def finish_updates() -> None:
        if table.updatesEnabled():
            return
        table.setUpdatesEnabled(True)

    def fill_batch() -> None:
        if not is_current():
            finish_updates()
            return

        start = state["next_row"]
        end = min(start + _PREVIEW_BATCH_SIZE, n_show)
        for row_idx in range(start, end):
            if not is_current():
                finish_updates()
                return
            for col_idx, col in enumerate(_PREVIEW_COLUMNS):
                item = QTableWidgetItem(_format_cell(col, columns[col][row_idx]))
                item.setFlags(flags)
                table.setItem(row_idx, col_idx, item)

        state["next_row"] = end
        if end >= n_show:
            finish_updates()
            _apply_preview_header_modes(table, n_rows=n_show)
            table.setSortingEnabled(False)
            return

        QTimer.singleShot(0, fill_batch)

    QTimer.singleShot(0, fill_batch)


def make_summary_label() -> QLabel:
    label = QLabel("No plan generated yet.")
    label.setWordWrap(True)
    return label
