"""Preview table for sigma auto-tuning."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

from scan_kit.common.devices_xml import IC_SIGMA_DEVICES

from .sigma_tune import SigmaTunePreviewRow

_ENERGY_COLUMN = "Energy (MeV)"
_VARIANCE_COLUMN = "Max σ² (mm²)"
_EXTREME_PCT_COLUMN = "Max ext. Δ (%)"
_IC_COLUMNS = IC_SIGMA_DEVICES
_TABLE_COLUMNS = (
    _ENERGY_COLUMN,
    *_IC_COLUMNS,
    _VARIANCE_COLUMN,
    _EXTREME_PCT_COLUMN,
)


def clear_sigma_preview_table(table: QTableWidget) -> None:
    table.clear()
    table.setRowCount(0)
    table.setColumnCount(len(_TABLE_COLUMNS))
    table.setHorizontalHeaderLabels(list(_TABLE_COLUMNS))
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


def fill_sigma_preview_table(
    table: QTableWidget,
    rows: list[SigmaTunePreviewRow] | None,
) -> None:
    clear_sigma_preview_table(table)
    if not rows:
        return

    grouped = _group_rows_by_energy(rows)
    table.setRowCount(len(grouped))
    variance_col = _TABLE_COLUMNS.index(_VARIANCE_COLUMN)
    extreme_col = _TABLE_COLUMNS.index(_EXTREME_PCT_COLUMN)
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
                    f"{device}: {entry.old_k0:.3f} mm → {entry.new_k0:.3f} mm "
                    f"(Δ {entry.delta_k0:+.3f} mm, σ² {entry.sigma_variance:.4f} mm², "
                    f"{entry.extreme_kind or 'extreme'} σ {entry.extreme_observed_mm:.3f} mm "
                    f"is {entry.extreme_pct_deviation:.1f}% from new, "
                    f"{entry.n_spots} spots)"
                )
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            table.setItem(row_idx, col_idx, item)

        max_variance = _max_band_variance(by_device)
        variance_item = QTableWidgetItem(_format_variance(max_variance))
        variance_item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        variance_item.setToolTip(_variance_tooltip(by_device))
        table.setItem(row_idx, variance_col, variance_item)

        max_extreme_pct, extreme_entry = _max_band_extreme_pct(by_device)
        extreme_item = QTableWidgetItem(_format_percent(max_extreme_pct))
        extreme_item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        extreme_item.setToolTip(_extreme_pct_tooltip(by_device, extreme_entry))
        table.setItem(row_idx, extreme_col, extreme_item)


def _group_rows_by_energy(
    rows: list[SigmaTunePreviewRow],
) -> list[tuple[str, dict[str, SigmaTunePreviewRow]]]:
    grouped: dict[tuple[float, float], dict[str, SigmaTunePreviewRow]] = {}
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


def _max_band_variance(by_device: dict[str, SigmaTunePreviewRow]) -> float:
    if not by_device:
        return float("nan")
    return max(entry.sigma_variance for entry in by_device.values())


def preview_energy_band_count(rows: list[SigmaTunePreviewRow]) -> int:
    return len(_group_rows_by_energy(rows))


def max_preview_extreme_pct_deviation(rows: list[SigmaTunePreviewRow]) -> float | None:
    """Largest per-energy ``Max ext. Δ (%)`` value across the preview table."""
    if not rows:
        return None
    grouped = _group_rows_by_energy(rows)
    values: list[float] = []
    for _, by_device in grouped:
        pct, _ = _max_band_extreme_pct(by_device)
        if pct == pct:
            values.append(pct)
    return max(values) if values else None


def _max_band_extreme_pct(
    by_device: dict[str, SigmaTunePreviewRow],
) -> tuple[float, SigmaTunePreviewRow | None]:
    if not by_device:
        return float("nan"), None
    entry = max(by_device.values(), key=lambda row: row.extreme_pct_deviation)
    return entry.extreme_pct_deviation, entry


def _variance_tooltip(by_device: dict[str, SigmaTunePreviewRow]) -> str:
    lines = [
        f"{device}: σ² = {entry.sigma_variance:.4f} mm²"
        for device, entry in by_device.items()
    ]
    return "Observed spot σ variance by IC:\n" + "\n".join(lines)


def _extreme_pct_tooltip(
    by_device: dict[str, SigmaTunePreviewRow],
    worst: SigmaTunePreviewRow | None,
) -> str:
    lines = [
        (
            f"{device}: {entry.extreme_kind} σ = {entry.extreme_observed_mm:.3f} mm, "
            f"{entry.extreme_pct_deviation:.1f}% from new {entry.new_k0:.3f} mm"
        )
        for device, entry in by_device.items()
    ]
    header = "Furthest min/max observed σ from new assignment by IC:"
    if worst is not None:
        header += (
            f"\n(worst: {worst.extreme_kind} σ {worst.extreme_observed_mm:.3f} mm "
            f"→ {worst.extreme_pct_deviation:.1f}%)"
        )
    return header + "\n" + "\n".join(lines)


def _format_energy_band(min_energy: float, max_energy: float) -> str:
    if abs(max_energy - min_energy) < 0.05:
        return f"{(min_energy + max_energy) / 2.0:.1f}"
    return f"{min_energy:.1f}–{max_energy:.1f}"


def _format_ic_cell(row: SigmaTunePreviewRow) -> str:
    old_s = _format_sigma(row.old_k0)
    new_s = _format_sigma(row.new_k0)
    if abs(row.delta_k0) < 1e-12:
        return old_s
    return f"{old_s} → {new_s}"


def _format_sigma(value: float) -> str:
    return f"{value:.3f}"


def _format_variance(value: float) -> str:
    if not (value == value):  # NaN
        return "—"
    return f"{value:.4f}"


def _format_percent(value: float) -> str:
    if not (value == value):  # NaN
        return "—"
    return f"{value:.1f}%"
