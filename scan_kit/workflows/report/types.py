"""Shared types for PDF report generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from scan_kit.common.session_meta import SessionMeta
from scan_kit.common.settings import ViewSettings
from scan_kit.views import ViewEntry


@dataclass
class ViewRenderResult:
    display_name: str
    module_name: str
    success: bool
    skip_reason: str | None = None


@dataclass
class ReportConfig:
    title: str
    subtitle: str
    author: str
    output_path: Path
    session_ids: list[str]
    base_dir: str
    settings: ViewSettings
    views: list[ViewEntry]
    session_meta: dict[str, SessionMeta | None] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
