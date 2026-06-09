"""Tests for beam error motion vs energy view."""

from __future__ import annotations

from pathlib import Path

from scan_kit.views.beam_motion_energy import _load_session_spill_paths

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"


def test_g2_session_resolves_all_energies() -> None:
    paths = _load_session_spill_paths("590658542", str(TEST_DATA))
    assert paths is not None
    assert len(paths) == 25


def test_g2_full_range_session_resolves_all_energies() -> None:
    paths = _load_session_spill_paths("883144654", str(TEST_DATA))
    assert paths is not None
    assert len(paths) == 76


def test_g3_session_resolves_all_energies() -> None:
    paths = _load_session_spill_paths("1091134775", str(TEST_DATA))
    assert paths is not None
    assert len(paths) == 76
