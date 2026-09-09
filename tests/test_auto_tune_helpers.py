"""Unit tests for auto-tune helpers, registry, and workflow validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from scan_kit.workflows.config_tuning.auto_tuning.registry import (
    AUTO_TUNE_REGISTRY,
    get_auto_tune_workflow,
)
from scan_kit.workflows.config_tuning.auto_tuning.session_params import parse_session_ids
from scan_kit.workflows.config_tuning.auto_tuning.sigma_tune import (
    DEFAULT_SIGMA_LOWER_HEADROOM_PERCENT,
    DEFAULT_SIGMA_TOLERANCE_PERCENT,
    format_sigma_k0,
    normalize_sigma_lower_headroom_percent,
    normalize_sigma_optimize_mode,
    normalize_sigma_tolerance_percent,
)
from scan_kit.workflows.config_tuning.auto_tuning.workflows.position_offset_tuning import (
    PositionOffsetTuningWorkflow,
)
from scan_kit.workflows.config_tuning.auto_tuning.workflows.sigma_tuning import (
    SigmaTuningWorkflow,
)

_ROOT = Path(__file__).resolve().parent.parent
_TEST_DATA = _ROOT / "test_data"
_SESSION = "1943968267"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"session_ids": ["a", "b", "a"]}, ["a", "b"]),
        ({"session_id": "legacy"}, ["legacy"]),
        ({"session_ids": [], "session_id": "fallback"}, []),
        ({}, []),
    ],
)
def test_parse_session_ids(raw: dict, expected: list[str]) -> None:
    assert parse_session_ids(raw) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("median", "median"),
        ("weighted_average", "weighted_average"),
        ("min_max_midpoint", "min_max_midpoint"),
        ("bogus", "median"),
        (None, "median"),
    ],
)
def test_normalize_sigma_optimize_mode(value: str | None, expected: str) -> None:
    assert normalize_sigma_optimize_mode(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, DEFAULT_SIGMA_TOLERANCE_PERCENT),
        (15.0, 15.0),
        ("12.5", 12.5),
        (-1.0, DEFAULT_SIGMA_TOLERANCE_PERCENT),
        ("not-a-number", DEFAULT_SIGMA_TOLERANCE_PERCENT),
        (float("nan"), DEFAULT_SIGMA_TOLERANCE_PERCENT),
    ],
)
def test_normalize_sigma_tolerance_percent(value, expected: float) -> None:
    assert normalize_sigma_tolerance_percent(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, DEFAULT_SIGMA_LOWER_HEADROOM_PERCENT),
        (0.0, 0.0),
        ("2.5", 2.5),
        (-0.1, DEFAULT_SIGMA_LOWER_HEADROOM_PERCENT),
        ("x", DEFAULT_SIGMA_LOWER_HEADROOM_PERCENT),
    ],
)
def test_normalize_sigma_lower_headroom_percent(value, expected: float) -> None:
    assert normalize_sigma_lower_headroom_percent(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2.588, "2.588"),
        (2.535e-08, "2.535000E-08"),
        (1496.737, "1.496737E+03"),
        (1000.0, "1.000000E+03"),
        (5.0, "5"),
    ],
)
def test_format_sigma_k0(value: float, expected: str) -> None:
    assert format_sigma_k0(value) == expected


def test_auto_tune_registry_contains_every_workflow() -> None:
    ids = {workflow.id for workflow in AUTO_TUNE_REGISTRY}
    assert ids == {"sigma_tuning", "position_offset_tuning", "ic_distance_tuning"}


def test_get_auto_tune_workflow_known_and_unknown() -> None:
    assert get_auto_tune_workflow("sigma_tuning") is not None
    assert get_auto_tune_workflow("position_offset_tuning") is not None
    assert get_auto_tune_workflow("ic_distance_tuning") is not None
    assert get_auto_tune_workflow("not_a_workflow") is None


def test_sigma_tuning_workflow_validate_rejects_missing_inputs(tmp_path: Path) -> None:
    workflow = SigmaTuningWorkflow()
    assert workflow.validate({"session_ids": [], "data_dir": ""})
    assert workflow.validate({"session_ids": [_SESSION], "data_dir": ""})

    missing_dir = tmp_path / "missing"
    errors = workflow.validate(
        {"session_ids": [_SESSION], "data_dir": str(missing_dir)}
    )
    assert errors
    assert "not a directory" in errors[0].lower()


def test_sigma_tuning_workflow_validate_accepts_fixture(tmp_path: Path) -> None:
    workflow = SigmaTuningWorkflow()
    assert not workflow.validate(
        {"session_ids": [_SESSION], "data_dir": str(_TEST_DATA)}
    )


def test_position_offset_workflow_validate_rejects_missing_sessions() -> None:
    workflow = PositionOffsetTuningWorkflow()
    errors = workflow.validate({"session_ids": [], "data_dir": str(_TEST_DATA)})
    assert errors
    assert "session" in errors[0].lower()


def test_sigma_tuning_workflow_apply_to_root_success() -> None:
    import xml.etree.ElementTree as ET

    devices_path = (
        _TEST_DATA / _SESSION / _SESSION / "config" / "map2map" / "devices.xml"
    )
    root = ET.fromstring(devices_path.read_text(encoding="utf-8"))
    workflow = SigmaTuningWorkflow()
    result = workflow.apply_to_root(
        root,
        {
            "session_ids": [_SESSION],
            "data_dir": str(_TEST_DATA),
            "sigma_tolerance_percent": 20.0,
            "sigma_lower_headroom_percent": 1.0,
        },
    )
    assert result.success
    assert result.sigma is not None
    assert result.sigma.bands_updated > 0
    assert "beam_sigma" in result.message


def test_sigma_tuning_workflow_apply_to_root_failure_keeps_tree_unchanged() -> None:
    import xml.etree.ElementTree as ET

    devices_path = (
        _TEST_DATA / _SESSION / _SESSION / "config" / "map2map" / "devices.xml"
    )
    before = devices_path.read_text(encoding="utf-8")
    root = ET.fromstring(before)
    workflow = SigmaTuningWorkflow()
    result = workflow.apply_to_root(
        root,
        {"session_ids": ["missing-session-id"], "data_dir": str(_TEST_DATA)},
    )
    assert not result.success
    assert ET.tostring(root, encoding="unicode") == ET.tostring(
        ET.fromstring(before),
        encoding="unicode",
    )
