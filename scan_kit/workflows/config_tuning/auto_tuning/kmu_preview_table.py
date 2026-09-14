"""Preview table for kMU auto-tuning: one row per ion-chamber device."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

from .kmu_tune import KmuTunePreviewRow
from .sigma_tune import format_sigma_k0

_TABLE_COLUMNS = (
    "Device",
    "Role",
    "Old K_MU",
    "New K_MU",
    "Δ K_MU",
    "Measured MU",
    "vs primary",
    "vs plan",
    "Write",
)

_LARGE_MOVE_COLOR = QColor(200, 120, 0)


def clear_kmu_preview_table(table: QTableWidget) -> None:
    table.clear()
    table.setRowCount(0)
    table.setColumnCount(len(_TABLE_COLUMNS))
    table.setHorizontalHeaderLabels(list(_TABLE_COLUMNS))
    header = table.horizontalHeader()
    header.setDefaultAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )
    header.setStretchLastSection(False)
    header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


def fill_kmu_preview_table(
    table: QTableWidget,
    rows: list[KmuTunePreviewRow] | None,
) -> None:
    clear_kmu_preview_table(table)
    if not rows:
        return

    table.setRowCount(len(rows))
    for row_idx, row in enumerate(rows):
        vs_primary = (
            f"{row.vs_primary_pct_before:+.2f}% → {row.vs_primary_pct_after:+.2f}%"
        )
        vs_plan = (
            f"{row.vs_plan_pct:+.2f}%"
            if row.vs_plan_pct == row.vs_plan_pct
            else "—"
        )
        delta = (
            f"{row.delta_pct:+.2f}%"
            if row.delta_pct == row.delta_pct
            else "—"
        )
        cells = (
            row.device,
            row.role,
            format_sigma_k0(row.old_kmu),
            format_sigma_k0(row.new_kmu),
            delta,
            f"{row.measured_mu:.4g}",
            vs_primary,
            vs_plan,
            "yes" if row.will_write else "no",
        )
        tooltip = _row_tooltip(row)
        highlight = row.is_large_scale or not row.will_write
        for col_idx, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            item.setToolTip(tooltip)
            if highlight:
                item.setForeground(QBrush(_LARGE_MOVE_COLOR))
            table.setItem(row_idx, col_idx, item)


def _row_tooltip(row: KmuTunePreviewRow) -> str:
    lines = [
        f"{row.device} ({row.family.upper()}, {row.role})",
        f"{row.n_spots} spots, measured {row.measured_mu:.4g} MU",
        f"K_MU {format_sigma_k0(row.old_kmu)} → {format_sigma_k0(row.new_kmu)} "
        f"(scale {row.scale:.5f})",
        f"vs primary {row.vs_primary_pct_before:+.2f}% → {row.vs_primary_pct_after:+.2f}%",
    ]
    if row.vs_plan_pct == row.vs_plan_pct:
        lines.append(f"vs plan {row.vs_plan_pct:+.2f}% (diagnostic; not a write target)")
    if row.max_energy_residual_pct == row.max_energy_residual_pct:
        lines.append(
            f"max per-energy residual vs global scale: {row.max_energy_residual_pct:.2f}%"
        )
    if row.is_large_scale:
        lines.append("Large K_MU scale for a dose calibration.")
    if not row.will_write:
        lines.append("This device will not be written.")
    return "\n".join(lines)


def preview_write_count(rows: list[KmuTunePreviewRow]) -> int:
    return sum(1 for row in rows if row.will_write)


def max_preview_kmu_delta_pct(rows: list[KmuTunePreviewRow]) -> float | None:
    values = [abs(row.delta_pct) for row in rows if row.delta_pct == row.delta_pct]
    return max(values) if values else None


def primary_measured_mu(rows: list[KmuTunePreviewRow]) -> float | None:
    for row in rows:
        if row.role == "primary":
            return row.measured_mu
    return None
