"""UI for viewing and verifying Pyramid ``.md5`` integrity sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QGroupBox, QLabel, QSizePolicy, QVBoxLayout, QWidget

from scan_kit.common.file_integrity import (
    IntegrityStatus,
    compute_hex_digest,
    format_mtime,
    parse_sidecar,
    pyramid_utc_mtime,
    sidecar_path,
    source_path_from_sidecar,
    status_label,
    verify_file_integrity,
)


@dataclass(frozen=True)
class IntegrityReport:
    """Verification snapshot for display."""

    data_path: Path
    sidecar_path: Path
    status: IntegrityStatus
    summary: str
    detail_lines: tuple[str, ...]


def build_integrity_report(
    data_path: Path,
    *,
    check_mtime: bool = True,
) -> IntegrityReport:
    """Build a human-readable integrity report for the config editor."""
    data_path = data_path.resolve()
    md5_path = sidecar_path(data_path)
    lines: list[str] = []
    result = verify_file_integrity(data_path, check_mtime=check_mtime)

    lines.append(f"Data file: {data_path.name}")
    lines.append(f"Sidecar: {md5_path.name}")

    if md5_path.is_file():
        raw = md5_path.read_bytes()
        lines.append(f"Sidecar size: {len(raw)} bytes")
        try:
            sidecar = parse_sidecar(md5_path)
            lines.append(f"Salt: {sidecar.salt}")
            lines.append(f"Stored digest: {sidecar.hex_digest}")
            file_data = data_path.read_bytes()
            computed = compute_hex_digest(file_data, sidecar.salt)
            lines.append(f"Computed digest: {computed}")
            hash_ok = computed == sidecar.hex_digest
            lines.append(f"Hash: {'OK' if hash_ok else 'MISMATCH'}")
            lines.append(f"Stored timestamp: {format_mtime(sidecar.stored_mtime)}")
            if data_path.is_file():
                actual_ts = pyramid_utc_mtime(data_path)
                lines.append(f"File timestamp: {format_mtime(actual_ts)}")
                ts_ok = abs(actual_ts - sidecar.stored_mtime) <= 1e-6
                lines.append(f"Timestamp: {'OK' if ts_ok else 'MISMATCH'}")
        except (OSError, ValueError) as exc:
            lines.append(f"Parse error: {exc}")
    else:
        lines.append("Sidecar: (missing)")

    if result.expected_digest is not None and result.sidecar is not None:
        lines.append(f"Expected digest: {result.expected_digest}")

    summary = status_label(result.status)
    return IntegrityReport(
        data_path=data_path,
        sidecar_path=md5_path,
        status=result.status,
        summary=summary,
        detail_lines=tuple(lines),
    )


def build_sidecar_only_report(sidecar_file: Path) -> IntegrityReport:
    """Report when the user opens the ``.md5`` file directly."""
    data_path = source_path_from_sidecar(sidecar_file).resolve()
    if data_path.is_file():
        return build_integrity_report(data_path)
    lines = (
        f"Sidecar: {Path(sidecar_file).name}",
        f"Paired data file: {data_path.name}",
        "Data file: (missing on disk)",
    )
    return IntegrityReport(
        data_path=data_path,
        sidecar_path=Path(sidecar_file).resolve(),
        status=IntegrityStatus.SOURCE_FILE_NOT_EXIST,
        summary=status_label(IntegrityStatus.SOURCE_FILE_NOT_EXIST),
        detail_lines=lines,
    )


def is_integrity_data_file(path: str | Path) -> bool:
    """True for map2map-style XML data files (not ``.xml.md5`` sidecars)."""
    name = Path(path).name.lower()
    return name.endswith(".xml") and not name.endswith(".xml.md5")


def integrity_passed(status: IntegrityStatus) -> bool:
    return status == IntegrityStatus.OK


def integrity_badge_markup(status: IntegrityStatus) -> tuple[str, str, str]:
    """Return (glyph, color, tooltip) for the compact header badge."""
    if integrity_passed(status):
        return ("✓", "#1a7f37", status_label(status))
    return ("✗", "#cf222e", status_label(status))


class FileIntegrityWidget(QGroupBox):
    """Full sidecar details — shown only when the user opens a ``.md5`` file."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("File integrity", parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinAndMaxSize)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout.addWidget(self._summary)

        self._details = QLabel()
        self._details.setWordWrap(True)
        self._details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._details.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._details.setFont(mono)
        layout.addWidget(self._details)

        self.hide()

    def set_report(self, report: IntegrityReport | None) -> None:
        if report is None:
            self.hide()
            return
        self.show()
        self._apply_status_style(report.status)
        self._summary.setText(report.summary)
        self._details.setText("\n".join(report.detail_lines))

    def _apply_status_style(self, status: IntegrityStatus) -> None:
        _, color, _ = integrity_badge_markup(status)
        self._summary.setStyleSheet(f"color: {color}; font-weight: 600;")
