"""Preview table for position-offset auto-tuning."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

from scan_kit.common.devices_xml import IC_SIGMA_DEVICES

from .position_offset_tune import PositionOffsetTunePreviewRow

_ENERGY_COLUMN = "Energy (MeV)"
_VARIANCE_COLUMN = "Max err² (mm²)"
_RESIDUAL_COLUMN = "Max residual (mm)"
_IC_COLUMNS = IC_SIGMA_DEVICES
_TABLE_COLUMNS = (
    _ENERGY_COLUMN,
    *_IC_COLUMNS,
    _VARIANCE_COLUMN,
    _RESIDUAL_COLUMN,
)


def clear_position_offset_preview_table(table: QTableWidget) -> None:
    table.clear()
    table.setRowCount(0)
    table.setColumnCount(len(_TABLE_COLUMNS))
    table.setHorizontalHeaderLabels(list(_TABLE_COLUMNS))
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


def fill_position_offset_preview_table(
    table: QTableWidget,
    rows: list[PositionOffsetTunePreviewRow] | None,
) -> None:
    clear_position_offset_preview_table(table)
    if not rows:
        return

    grouped = _group_rows_by_energy(rows)
    table.setRowCount(len(grouped))
    variance_col = _TABLE_COLUMNS.index(_VARIANCE_COLUMN)
    residual_col = _TABLE_COLUMNS.index(_RESIDUAL_COLUMN)
    for row_idx, (energy_label, by_device) in enumerate(grouped):
        energy_item = QTableWidgetItem(energy_label)
        energy_item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        table.setItem(row_idx, 0, energy_item)

        for col_idx, device in enumerate(_IC_COLUMNS, start=1):
            entry = by_device.get(device)
            if entry is None:
                item = QTableWidgetItem("—")
            else:
                item = QTableWidgetItem(_format_ic_cell(entry))
                item.setToolTip(
                    f"{device}: {entry.old_offset:.3f} mm → {entry.new_offset:.3f} mm "
                    f"(Δ {entry.delta_offset:+.3f} mm, band median err "
                    f"{entry.band_median_error:+.3f} mm, err² "
                    f"{entry.error_variance:.4f} → {entry.residual_variance:.4f} mm², "
                    f"max |err| {entry.max_abs_error_mm:.3f} → "
                    f"{entry.max_residual_mm:.3f} mm, "
                    f"{entry.n_samples} samples)"
                )
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            table.setItem(row_idx, col_idx, item)

        max_variance_before, max_variance_after = _max_band_variance(by_device)
        variance_item = QTableWidgetItem(
            _format_before_after(
                max_variance_before,
                max_variance_after,
                formatter=_format_variance,
            )
        )
        variance_item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        variance_item.setToolTip(_variance_tooltip(by_device))
        table.setItem(row_idx, variance_col, variance_item)

        max_error_before, max_error_after, worst = _max_band_abs_error(by_device)
        residual_item = QTableWidgetItem(
            _format_before_after(
                max_error_before,
                max_error_after,
                formatter=_format_mm,
            )
        )
        residual_item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        residual_item.setToolTip(_residual_tooltip(by_device, worst))
        table.setItem(row_idx, residual_col, residual_item)


def _group_rows_by_energy(
    rows: list[PositionOffsetTunePreviewRow],
) -> list[tuple[str, dict[str, PositionOffsetTunePreviewRow]]]:
    grouped: dict[tuple[float, float], dict[str, PositionOffsetTunePreviewRow]] = {}
    for row in rows:
        key = (row.min_energy, row.max_energy)
        grouped.setdefault(key, {})[row.device] = row

    ordered_keys = sorted(
        grouped.keys(),
        key=lambda band: -((band[0] + band[1]) / 2.0),
    )
    return [
        (_format_energy_band(min_e, max_e), grouped[(min_e, max_e)])
        for min_e, max_e in ordered_keys
    ]


def _max_band_variance(
    by_device: dict[str, PositionOffsetTunePreviewRow],
) -> tuple[float, float]:
    if not by_device:
        return float("nan"), float("nan")
    return (
        max(entry.error_variance for entry in by_device.values()),
        max(entry.residual_variance for entry in by_device.values()),
    )


def preview_energy_band_count(rows: list[PositionOffsetTunePreviewRow]) -> int:
    return len(_group_rows_by_energy(rows))


def max_preview_residual_mm(rows: list[PositionOffsetTunePreviewRow]) -> float | None:
    """Largest per-energy max residual value across the preview table."""
    if not rows:
        return None
    grouped = _group_rows_by_energy(rows)
    values: list[float] = []
    for _, by_device in grouped:
        _, after, _ = _max_band_abs_error(by_device)
        if after == after:
            values.append(after)
    return max(values) if values else None


def _max_band_abs_error(
    by_device: dict[str, PositionOffsetTunePreviewRow],
) -> tuple[float, float, PositionOffsetTunePreviewRow | None]:
    if not by_device:
        return float("nan"), float("nan"), None
    before = max(entry.max_abs_error_mm for entry in by_device.values())
    worst = max(by_device.values(), key=lambda row: row.max_residual_mm)
    return before, worst.max_residual_mm, worst


def _variance_tooltip(by_device: dict[str, PositionOffsetTunePreviewRow]) -> str:
    lines = [
        (
            f"{device}: err² {entry.error_variance:.4f} → "
            f"{entry.residual_variance:.4f} mm²"
        )
        for device, entry in by_device.items()
    ]
    return "Position error variance by IC:\n" + "\n".join(lines)


def _residual_tooltip(
    by_device: dict[str, PositionOffsetTunePreviewRow],
    worst: PositionOffsetTunePreviewRow | None,
) -> str:
    lines = [
        (
            f"{device}: max |err| {entry.max_abs_error_mm:.3f} → "
            f"{entry.max_residual_mm:.3f} mm"
        )
        for device, entry in by_device.items()
    ]
    header = "Max absolute error → residual by IC:"
    if worst is not None:
        header += (
            f"\n(worst post-tune: {worst.device} "
            f"{worst.max_residual_mm:.3f} mm)"
        )
    return header + "\n" + "\n".join(lines)


def _format_energy_band(min_energy: float, max_energy: float) -> str:
    if abs(max_energy - min_energy) < 0.05:
        return f"{(min_energy + max_energy) / 2.0:.1f}"
    return f"{min_energy:.1f}–{max_energy:.1f}"


def _format_ic_cell(row: PositionOffsetTunePreviewRow) -> str:
    return _format_before_after(row.old_offset, row.new_offset, formatter=_format_mm)


def _format_before_after(
    before: float,
    after: float,
    *,
    formatter,
    tolerance: float = 1e-12,
) -> str:
    before_s = formatter(before)
    after_s = formatter(after)
    if not (before == before) or not (after == after):
        return "—"
    if abs(before - after) < tolerance:
        return before_s
    return f"{before_s} → {after_s}"


def _format_mm(value: float) -> str:
    if not (value == value):
        return "—"
    return f"{value:.3f}"


def _format_variance(value: float) -> str:
    if not (value == value):
        return "—"
    return f"{value:.4f}"
