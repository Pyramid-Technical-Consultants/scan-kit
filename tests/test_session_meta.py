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
    assert meta.map_extent_mm is None
    assert meta.layer_count is None
    assert meta.short_extent == "?"
    assert meta.short_layers == "?"


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


_GEOM_SAMPLE = """\
Date: Tue Aug 25 21:35:22 2026
Layer delivery: 27/27
Spot extent width: 118.75 mm
Spot extent height: 110.592 mm
Configuration name: working_no_hvttt_new_cal
"""


def test_parse_spot_extent_uses_larger_axis() -> None:
    meta = parse_termination_summary_text(_GEOM_SAMPLE)
    assert meta.map_extent_mm == 118.75
    assert meta.short_extent == "119"
    assert meta.layer_count == 27
    assert meta.short_layers == "27"


def test_zero_mm_extent_is_zero_not_missing() -> None:
    meta = parse_termination_summary_text(
        "Spot extent width: 0 mm\nSpot extent height: 0 mm\n"
    )
    assert meta.map_extent_mm == 0.0
    assert meta.short_extent == "0"


def test_layer_delivery_uses_planned_count() -> None:
    meta = parse_termination_summary_text("Layer delivery: 56/76\n")
    assert meta.layer_count == 76


def test_layer_delivery_without_slash() -> None:
    meta = parse_termination_summary_text("Layer delivery: 18\n")
    assert meta.layer_count == 18


def test_missing_geom_lines_are_none() -> None:
    meta = parse_termination_summary_text("Primary total dose: 1.0\n")
    assert meta.map_extent_mm is None
    assert meta.layer_count is None
    assert meta.short_extent == "?"
    assert meta.short_layers == "?"
