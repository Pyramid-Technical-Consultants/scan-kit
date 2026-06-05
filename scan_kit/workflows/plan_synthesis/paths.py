"""Default save locations for plan synthesis exports."""

from __future__ import annotations

from pathlib import Path

from scan_kit.workflows.report.paths import resolve_report_save_dir


def resolve_plan_synthesis_save_dir(last_save_dir: str | None = None) -> Path:
    """Return the last-used plan synthesis folder, else Downloads."""
    return resolve_report_save_dir(last_save_dir)
