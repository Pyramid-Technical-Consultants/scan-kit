"""Tests for session log parsing."""

from __future__ import annotations

from pathlib import Path

from scan_kit.common.session_log import (
    is_noise_message,
    load_session_log,
    message_template,
    parse_session_log_text,
)

_SAMPLE = """\
2026-04-09 20:39:04,214 DEBUG [dcs_log] START MAP FOR LAYER: 0 - TIMELINE(Start Map -> map Started from RoomController): T=1.42104s
2026-04-09 20:39:10,455 DEBUG [dcs_log] SCAN EXECUTING FOR LAYER: 0 - TIMELINE(Start Dosing -> Map Completed): T=6.24022s
2026-04-09 20:38:39,089 ERROR [dcs_log] RCI wdt read cnt: 213 - write counter: 214
2026-04-09 20:38:39,067 DEBUG [dcs_log] Got ACK to eCMD_GET_BEAM_OFFSETS
"""


def test_is_noise_message() -> None:
    assert is_noise_message("Got ACK to eCMD_GET_BEAM_OFFSETS")
    assert not is_noise_message("SCAN EXECUTING FOR LAYER: 0")


def test_message_template_normalizes_numbers_and_paths() -> None:
    msg = 'Controller: KX8 fail to get file: "/var/log/ptc_ex/1091134775/config/nozzle/KX8/prescription.json"'
    assert "<path>" in message_template(msg)
    assert "1091134775" not in message_template(msg)


def test_parse_timeline_and_wdt() -> None:
    data = parse_session_log_text(_SAMPLE, session_id="test")
    row = data.layer_timeline[0]
    assert row.start_map_s == 1.42104
    assert row.scan_execute_s == 6.24022
    assert data.wdt_mismatches["RCI"] == 1
    assert data.error_count == 1
    assert is_noise_message(data.entries[-1].message)


def test_load_fixture_session_log() -> None:
    root = Path(__file__).resolve().parent.parent
    base = root / "test_data"
    data = load_session_log("1091134775", base)
    assert data is not None
    assert len(data.entries) > 1000
    assert data.layers_scanned >= 1
