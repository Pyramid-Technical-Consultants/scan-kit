"""Preview table for IC distance auto-tuning: one row per ion chamber."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

from .ic_distance_tune import IcDistanceTunePreviewRow

_TABLE_COLUMNS = (
    "IC",
    "Distance (mm)",
    "Δ distance",
    "Zero offset (mm)",
    "Systematic (mm)",
    "RMS err (mm)",
    "Max |err| (mm)",
    "Samples",
)

_LARGE_MOVE_COLOR = QColor(200, 120, 0)


def clear_ic_distance_preview_table(table: QTableWidget) -> None:
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


def fill_ic_distance_preview_table(
    table: QTableWidget,
    rows: list[IcDistanceTunePreviewRow] | None,
) -> None:
    clear_ic_distance_preview_table(table)
    if not rows:
        return

    table.setRowCount(len(rows))
    for row_idx, row in enumerate(rows):
        cells = (
            row.device,
            f"{row.old_sdd_mm:.3f} → {row.new_sdd_mm:.3f}",
            f"{row.delta_sdd_mm:+.2f} ± {row.sdd_stderr_mm:.2f} "
            f"({row.delta_sdd_percent:+.2f}%)",
            f"{row.old_offset_mm:.3f} → {row.new_offset_mm:.3f}",
            f"{row.systematic_removed_mm:.3f}",
            f"{row.rms_before_mm:.3f} → {row.rms_after_mm:.3f}",
            f"{row.max_abs_before_mm:.3f} → {row.max_abs_after_mm:.3f}",
            str(row.n_samples)
            if not row.n_rejected
            else f"{row.n_samples} (−{row.n_rejected})",
        )
        tooltip = _row_tooltip(row)
        for col_idx, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            item.setToolTip(tooltip)
            if row.is_large_move or row.is_within_noise:
                item.setForeground(QBrush(_LARGE_MOVE_COLOR))
            table.setItem(row_idx, col_idx, item)


def _row_tooltip(row: IcDistanceTunePreviewRow) -> str:
    lines = [
        f"{row.device}: fitted gain {row.gain:.5f} ± {row.gain_stderr:.5f} "
        f"({row.gain_significance:.1f}σ from no change)",
        f"plan span {row.plan_span_mm:.1f} mm over {row.n_samples} spots"
        + (
            f" ({row.n_rejected} rejected as outliers before fitting, still counted in "
            f"the residuals below)"
            if row.n_rejected
            else ""
        ),
        f"distance {row.delta_sdd_mm:+.2f} ± {row.sdd_stderr_mm:.2f} mm",
        f"zero offset {row.old_offset_mm:+.3f} → {row.new_offset_mm:+.3f} mm "
        f"(Δ {row.delta_offset_mm:+.3f} mm)",
        f"systematic removed: {row.systematic_removed_mm:.3f} mm of position error at "
        f"the worst edge of the field",
        f"residual RMS {row.rms_before_mm:.3f} → {row.rms_after_mm:.3f} mm",
        f"max |err| {row.max_abs_before_mm:.3f} → {row.max_abs_after_mm:.3f} mm",
    ]
    if row.max_abs_worsened:
        lines.append(
            "Max |err| rises because it is one spot: the fit minimises RMS, and "
            "removing a systematic of opposite sign unmasks an outlier it was hiding. "
            "Judge this change on systematic and RMS."
        )
    if row.is_within_noise:
        lines.append(
            "Change is within the fit's own uncertainty — per-spot scatter alone can "
            "produce it."
        )
    if row.is_large_move:
        lines.append(
            "Large move for a surveyed distance — check magnet calibration and "
            "strip pitch first."
        )
    return "\n".join(lines)


def preview_chamber_count(rows: list[IcDistanceTunePreviewRow]) -> int:
    return len(rows)


def max_preview_sdd_percent(rows: list[IcDistanceTunePreviewRow]) -> float | None:
    """Largest absolute distance change in the preview, as a percentage."""
    values = [
        abs(row.delta_sdd_percent)
        for row in rows
        if row.delta_sdd_percent == row.delta_sdd_percent
    ]
    return max(values) if values else None


def max_preview_systematic_mm(rows: list[IcDistanceTunePreviewRow]) -> float | None:
    """Largest position error the proposed change removes at a field edge."""
    return max((row.systematic_removed_mm for row in rows), default=None)
