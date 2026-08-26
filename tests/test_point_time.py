"""Tests for layer-run point_time loading."""

from __future__ import annotations

from scan_kit.common.session_source import (
    load_session_point_time_table,
    resolve_session_source,
)
from tests.conftest import G3_SESSION, TEST_DATA


def test_load_session_point_time_table_g3() -> None:
    src = resolve_session_source(G3_SESSION, str(TEST_DATA))
    assert src is not None
    table = load_session_point_time_table(src)
    assert table is not None
    assert len(table) > 0
    assert "point_time_ms" in table.columns
    assert table["point_time_ms"].notna().any()
