"""Tests for unified view option labeling."""

from __future__ import annotations

from scan_kit.data.types import (
    DATA_SOURCE_SPOT_CHAMBER,
    DATA_SOURCE_SPOT_ISO,
    DATA_SOURCE_TIMESLICE_CHAMBER,
    DATA_SOURCE_TIMESLICE_ISO,
)
from scan_kit.views.binned_summary_catalog import VIEW_OPTIONS as BINNED_OPTIONS
from scan_kit.views.distribution_catalog import VIEW_OPTIONS as DISTRIBUTION_OPTIONS
from scan_kit.views.unified_catalog import format_view_option_label


def test_format_view_option_label_suffix_only_when_both_frames() -> None:
    label = format_view_option_label(
        "Sigma (mm)",
        DATA_SOURCE_SPOT_CHAMBER,
        sibling_sources=(
            DATA_SOURCE_SPOT_ISO,
            DATA_SOURCE_SPOT_CHAMBER,
        ),
    )
    assert label == "Sigma (mm) (Chamber)"

    iso_only = format_view_option_label(
        "Dose Ratios",
        DATA_SOURCE_SPOT_ISO,
        sibling_sources=(DATA_SOURCE_SPOT_ISO,),
    )
    assert iso_only == "Dose Ratios"


def test_binned_sigma_options_include_frame_labels() -> None:
    sigma_labels = {
        opt.label
        for opt in BINNED_OPTIONS
        if opt.id == "sigma"
    }
    assert "Sigma (mm) (Isocenter)" in sigma_labels
    assert "Sigma (mm) (Chamber)" in sigma_labels


def test_binned_position_error_spot_has_chamber_variant() -> None:
    sources = {
        opt.source
        for opt in BINNED_OPTIONS
        if opt.id == "position_error"
        and "Chamber" in opt.label
    }
    assert DATA_SOURCE_SPOT_CHAMBER in sources


def test_binned_sigma_error_matches_distribution_sources() -> None:
    dist_sources = {
        opt.source
        for opt in DISTRIBUTION_OPTIONS
        if opt.id == "sigma_error"
    }
    binned_sources = {
        opt.source
        for opt in BINNED_OPTIONS
        if opt.id == "sigma_error"
    }
    assert binned_sources == dist_sources


def test_binned_ic12_timeslice_has_chamber_variant() -> None:
    sources = {
        opt.source
        for opt in BINNED_OPTIONS
        if opt.id == "ic12_pos_diff"
        and "Chamber" in opt.label
    }
    assert DATA_SOURCE_TIMESLICE_CHAMBER in sources


def test_distribution_position_spot_has_iso_and_chamber_rows() -> None:
    labels = {
        opt.label
        for opt in DISTRIBUTION_OPTIONS
        if opt.id == "position" and opt.source in (
            DATA_SOURCE_SPOT_ISO,
            DATA_SOURCE_SPOT_CHAMBER,
        )
    }
    assert any("Isocenter" in label for label in labels)
    assert any("Chamber" in label for label in labels)
