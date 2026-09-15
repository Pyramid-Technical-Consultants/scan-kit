"""Parse termination_summary.txt including Configuration name."""

from __future__ import annotations

from datetime import datetime

from scan_kit.common.session_meta import parse_termination_summary_text


_SAMPLE = """\
Termination status: eNORMAL
Date: Thu Sep 10 21:07:41 2026
Session ID: 1584882828
Error code: 0
Room number: 4
Primary total dose: 38.0223 MU
Treatment time: 251.604 seconds
Configuration name: working_hvtt_new_cal_gates_tuned
"""


def test_parse_configuration_name() -> None:
    meta = parse_termination_summary_text(_SAMPLE)
    assert meta.config_name == "working_hvtt_new_cal_gates_tuned"
    assert meta.short_config == "working_hvtt_new_cal_gates_tuned"
    assert meta.room_number == 4
    assert meta.primary_mu == 38.0223
    assert meta.treatment_time_s == 251
    assert meta.date == datetime(2026, 9, 10, 21, 7, 41)


def test_missing_configuration_name_is_question_mark() -> None:
    meta = parse_termination_summary_text(
        "Date: Thu Sep 10 21:07:41 2026\nPrimary total dose: 1.0\n"
    )
    assert meta.config_name is None
    assert meta.short_config == "?"
    assert meta.room_number is None
    assert meta.short_room == "?"


def test_blank_configuration_name_is_question_mark() -> None:
    meta = parse_termination_summary_text("Configuration name:   \n")
    assert meta.config_name is None
    assert meta.short_config == "?"
