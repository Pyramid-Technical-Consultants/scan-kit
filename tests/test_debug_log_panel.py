"""Tests for the launcher debug log panel helpers."""

from __future__ import annotations

from datetime import datetime
from io import StringIO

from scan_kit.common.debug_log_panel import (
    _StreamTee,
    format_log_line,
)


def test_format_log_line_includes_level_source_and_message() -> None:
    when = datetime(2026, 6, 12, 14, 30, 1)
    line = format_log_line(
        level="ERROR",
        source="scan_kit.views.foo",
        message="something broke",
        now=when,
    )
    assert line == "14:30:01 [ERROR] [scan_kit.views.foo] something broke"


def test_stream_tee_forwards_complete_lines() -> None:
    original = StringIO()
    captured: list[tuple[str, str, str]] = []

    tee = _StreamTee(
        original,
        level="STDERR",
        source="stderr",
        emit_fn=lambda level, source, message: captured.append((level, source, message)),
    )
    tee.write("first line\nsecond")
    tee.write(" line\n")
    tee.flush()

    assert original.getvalue() == "first line\nsecond line\n"
    assert captured == [
        ("STDERR", "stderr", "first line"),
        ("STDERR", "stderr", "second line"),
    ]
