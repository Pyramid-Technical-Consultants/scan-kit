"""Tests for IGX key helpers and blob decoding."""

from scan_kit.igx.blobs import unwrap_field_update
from scan_kit.igx.keys import field_subscribe_key


def test_field_subscribe_key_adds_value_suffix() -> None:
    assert field_subscribe_key("admin/version") == "/admin/version/value"
    assert field_subscribe_key("/admin/version/value") == "/admin/version/value"


def test_unwrap_field_update_scalar_pair() -> None:
    assert unwrap_field_update([42, 1.0]) == 42


def test_unwrap_field_update_history_list() -> None:
    assert unwrap_field_update([[10, 1.0], [20, 2.0]]) == 20
