"""Tests for PDF report generation."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest

from scan_kit.common.settings import ViewSettings
from scan_kit.views import VIEWS
from scan_kit.workflows.report import (
    REPORT_EXCLUDED_MODULES,
    report_view_groups,
)
from scan_kit.workflows.report.builder import build_report_pdf
from scan_kit.workflows.report.naming import suggest_report_filename
from scan_kit.workflows.report.paths import resolve_report_save_dir
from scan_kit.workflows.report.types import ReportConfig

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEST_DATA = _PROJECT_ROOT / "test_data"


def _pdf_page_count(path: Path) -> int:
    data = path.read_bytes().decode("latin-1", errors="ignore")
    return len(re.findall(r"/Type\s*/Page\b", data))


def _first_session_id() -> str | None:
    if not _TEST_DATA.is_dir():
        return None
    for child in sorted(_TEST_DATA.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            return child.name
    return None


def test_suggest_report_filename_uses_notes_and_views() -> None:
    name = suggest_report_filename(
        ["1943968267"],
        {"1943968267": "Calibration Check"},
        [("Dose Ratios vs Energy", "dose_ratios_energy")],
        generated_at=datetime(2026, 6, 5, 14, 30, 0),
    )
    assert name.endswith(".pdf")
    assert name.startswith("scan-kit-2026-06-05_")
    assert "calibration-check" in name
    assert "dose-ratios-energy" in name
    assert len(name) <= 100


def test_suggest_report_filename_collapses_many_views() -> None:
    views = [
        ("A", "dose_ratios_energy"),
        ("B", "position_error_energy"),
        ("C", "sigma_energy"),
        ("D", "current_ratios"),
    ]
    name = suggest_report_filename(["s1", "s2"], {}, views)
    assert name.endswith(".pdf")
    assert "4-views" in name
    assert "2-sessions" in name


def test_resolve_report_save_dir_prefers_last_saved() -> None:
    last = str(_PROJECT_ROOT)
    assert resolve_report_save_dir(last) == _PROJECT_ROOT


def test_report_view_groups_excludes_non_static_modules() -> None:
    reportable_modules = {
        module_name for _title, entries in report_view_groups() for _name, module_name in entries
    }
    all_modules = {module_name for _name, module_name in VIEWS}
    assert REPORT_EXCLUDED_MODULES.issubset(all_modules - reportable_modules)
    assert "ic_timeslice_replay" not in reportable_modules
    assert "session_log_compare" not in reportable_modules
    assert "ic_audio_export" not in reportable_modules
    assert "dose_ratios_energy" in reportable_modules


def test_build_report_pdf_smoke(tmp_path: Path) -> None:
    session_id = _first_session_id()
    if session_id is None:
        pytest.skip("test_data session folder not available")

    output_path = tmp_path / "report-smoke.pdf"
    config = ReportConfig(
        title="Smoke Test Report",
        subtitle=f"Sessions: {session_id}",
        author="pytest",
        output_path=output_path,
        session_ids=[session_id],
        base_dir=str(_TEST_DATA),
        settings=ViewSettings(),
        views=[
            ("Dose Ratios vs Energy", "dose_ratios_energy"),
            ("Position Error vs Energy", "position_error_energy"),
        ],
        generated_at=datetime(2026, 6, 5, 12, 0, 0),
    )

    path, results = build_report_pdf(config)

    assert path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes()[:4] == b"%PDF"
    assert _pdf_page_count(output_path) >= 3
    assert any(result.success for result in results)
