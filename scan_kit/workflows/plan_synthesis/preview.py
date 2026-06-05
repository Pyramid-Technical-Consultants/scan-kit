"""Preview helpers for generated input_map plans."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHeaderView, QLabel, QTableWidget, QTableWidgetItem

from .input_map import INPUT_MAP_EXPORT_COLUMNS, input_map_export_frame, plan_summary

PREVIEW_ROW_CAP = 5_000
_PREVIEW_BATCH_SIZE = 200
_HEADER_COLUMN_PADDING_PX = 20
DEFAULT_DELIVERY_RATE_MU_PER_S = 0.4


def estimate_delivery_seconds(
    total_mu: float,
    *,
    rate_mu_per_s: float = DEFAULT_DELIVERY_RATE_MU_PER_S,
) -> float:
    if rate_mu_per_s <= 0:
        return 0.0
    return total_mu / rate_mu_per_s


def format_delivery_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    total_minutes = int(seconds // 60)
    if total_minutes < 60:
        rem_s = int(round(seconds % 60))
        if rem_s == 0:
            return f"{total_minutes} min"
        return f"{total_minutes} min {rem_s} s"
    hours = total_minutes // 60
    minutes = total_minutes % 60
    if minutes == 0:
        return f"{hours} h"
    return f"{hours} h {minutes} min"


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
    delivery = format_delivery_duration(estimate_delivery_seconds(total_mu))
    msg = (
        f"{n_layers} {layer_word} · {n_spots} {spot_word} · "
        f"{total_mu:.3g} MU total · est. {delivery} delivery"
    )
    if n_spots > preview_row_cap:
        msg += f" · preview shows first {preview_row_cap:,} rows"
    return msg


def _format_cell(col: str, val: object) -> str:
    if col == "CHARGE_REQ(MU)":
        return f"{float(val):.4f}"
    if col in (
        "ENERGY(MeV)",
        "CURRENT(A)",
        "BEAM_SIZE(mm)",
        "X_POSITION(mm)",
        "Y_POSITION(mm)",
        "VELOCITY(mm/s)",
    ):
        return f"{float(val):g}"
    if col == "#NO":
        return str(int(val))
    return str(val)


def clear_preview_table(table: QTableWidget) -> None:
    """Drop all preview rows and release table item memory."""
    table.setSortingEnabled(False)
    table.clearContents()
    table.setRowCount(0)
    table.setColumnCount(len(INPUT_MAP_EXPORT_COLUMNS))
    table.setHorizontalHeaderLabels(list(INPUT_MAP_EXPORT_COLUMNS))
    _resize_preview_columns_to_headers(table)


def _resize_preview_columns_to_headers(table: QTableWidget) -> None:
    """Size each preview column to fit its header label."""
    hh = table.horizontalHeader()
    hh.setStretchLastSection(False)
    fm = hh.fontMetrics()
    for col_idx, label in enumerate(INPUT_MAP_EXPORT_COLUMNS):
        width = fm.horizontalAdvance(label) + _HEADER_COLUMN_PADDING_PX
        hh.setSectionResizeMode(col_idx, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(col_idx, width)


def fill_preview_table(table: QTableWidget, df: pd.DataFrame | None) -> int:
    """Synchronously populate *table* (small/empty previews only)."""
    if df is None or df.empty:
        clear_preview_table(table)
        return 0

    n_show = min(len(df), PREVIEW_ROW_CAP)
    preview = input_map_export_frame(df).iloc[:n_show]

    table.setSortingEnabled(False)
    table.clearContents()
    table.setColumnCount(len(INPUT_MAP_EXPORT_COLUMNS))
    table.setHorizontalHeaderLabels(list(INPUT_MAP_EXPORT_COLUMNS))
    table.setRowCount(n_show)

    flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
    table.setUpdatesEnabled(False)
    try:
        for col_idx, col in enumerate(INPUT_MAP_EXPORT_COLUMNS):
            column = preview[col].values
            for row_idx in range(n_show):
                item = QTableWidgetItem(_format_cell(col, column[row_idx]))
                item.setFlags(flags)
                table.setItem(row_idx, col_idx, item)
    finally:
        table.setUpdatesEnabled(True)

    _resize_preview_columns_to_headers(table)
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
    preview = input_map_export_frame(df).iloc[:n_show]
    columns = {col: preview[col].values for col in INPUT_MAP_EXPORT_COLUMNS}

    table.setSortingEnabled(False)
    table.clearContents()
    table.setColumnCount(len(INPUT_MAP_EXPORT_COLUMNS))
    table.setHorizontalHeaderLabels(list(INPUT_MAP_EXPORT_COLUMNS))
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
            for col_idx, col in enumerate(INPUT_MAP_EXPORT_COLUMNS):
                item = QTableWidgetItem(_format_cell(col, columns[col][row_idx]))
                item.setFlags(flags)
                table.setItem(row_idx, col_idx, item)

        state["next_row"] = end
        if end >= n_show:
            finish_updates()
            _resize_preview_columns_to_headers(table)
            table.setSortingEnabled(False)
            return

        QTimer.singleShot(0, fill_batch)

    QTimer.singleShot(0, fill_batch)


def make_summary_label() -> QLabel:
    label = QLabel("No plan generated yet.")
    label.setWordWrap(True)
    return label
