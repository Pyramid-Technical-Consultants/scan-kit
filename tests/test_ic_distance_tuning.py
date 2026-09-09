"""Tests for the joint IC distance + zero offset fit."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from scan_kit.common.devices_xml import IcGeometry
from scan_kit.common.session_position import (
    MeasuredPositionErrors,
    load_measured_position_errors_spot,
    merge_measured_position_errors,
)
from scan_kit.workflows.config_tuning.auto_tuning.ic_distance_tune import (
    MAX_ABS_GAIN_DEVIATION,
    MIN_PLAN_SPAN_MM,
    REJECTED_FRACTION_WARN,
    apply_ic_distances_to_tree,
    collect_distance_rows,
    fit_position_scale,
    format_distance_mm,
    physicality_warnings,
)

_TEST_DATA = Path(__file__).resolve().parent.parent / "test_data"
_VIRTUAL_SAD = 2500.0
_TRUE_SDD = 1200.0
_TRUE_OFFSET = 0.75
_CONFIG_SDD = 1250.0
_CONFIG_OFFSET = -1.5


def _devices_xml(sdd_mm: float = _CONFIG_SDD, offset_mm: float = _CONFIG_OFFSET) -> str:
    chambers = "".join(
        f'<ion_chamber><device name="{name}"/>'
        "<strip_count>128</strip_count><strip_to_mm>2</strip_to_mm>"
        f"<zero_offset_at_iso_mm>{offset_mm}</zero_offset_at_iso_mm>"
        f"<source_to_device_distance_mm>{sdd_mm}</source_to_device_distance_mm>"
        "<source_to_axis_distance_mm>999</source_to_axis_distance_mm>"
        "</ion_chamber>"
        for name in ("IC_1_X", "IC_1_Y", "IC_2_X", "IC_2_Y")
    )
    return f'<?xml version="1.0"?><devices>{chambers}</devices>'


def _chamber(sdd_mm: float, offset_mm: float) -> IcGeometry:
    return IcGeometry(
        name="IC_1_X",
        strip_count=128.0,
        strip_to_mm=2.0,
        zero_offset_at_iso_mm=offset_mm,
        reverse_strips=False,
        sdd_mm=sdd_mm,
        xml_sad_mm=999.0,
    )


def _delivered_strips(plan: np.ndarray) -> np.ndarray:
    """Strips the beam physically lands on to put spots at *plan*, per the true geometry."""
    truth = _chamber(_TRUE_SDD, _TRUE_OFFSET)
    return (plan - _TRUE_OFFSET) / truth.iso_mm_per_strip(_VIRTUAL_SAD) + truth.center_strip


def _reported(strips: np.ndarray, sdd_mm: float, offset_mm: float) -> np.ndarray:
    """What a session CSV holds: those strips converted by the config's geometry."""
    chamber = _chamber(sdd_mm, offset_mm)
    return np.asarray(chamber.strip_to_iso(strips, _VIRTUAL_SAD), dtype=float)


def _delivered_positions(plan: np.ndarray) -> np.ndarray:
    """Measured positions from a mis-calibrated devices.xml."""
    return _reported(_delivered_strips(plan), _CONFIG_SDD, _CONFIG_OFFSET)


def _measured(plan: np.ndarray, measured: np.ndarray) -> MeasuredPositionErrors:
    devices = ("IC_1_X", "IC_1_Y", "IC_2_X", "IC_2_Y")
    energies = np.full(plan.shape, 150.0)
    return MeasuredPositionErrors(
        by_device={d: (energies, measured - plan) for d in devices},
        plan_by_device={d: plan for d in devices},
    )


def test_fit_recovers_the_true_distance_and_offset() -> None:
    plan = np.linspace(-100.0, 100.0, 41)
    fit = fit_position_scale(plan, _delivered_positions(plan))
    assert fit is not None
    # sdd_new = sdd_old * gain, off_new = (off_old - bias) / gain.
    assert abs(_CONFIG_SDD * fit.gain - _TRUE_SDD) < 1e-6
    assert abs((_CONFIG_OFFSET - fit.bias) / fit.gain - _TRUE_OFFSET) < 1e-9
    assert fit.rms_after_mm < 1e-9
    assert fit.rms_before_mm > 1.0


def test_scatter_at_fixed_plan_positions_does_not_fake_a_scale_error() -> None:
    """Regression: fitting plan on measurement attenuates the gain by regression
    dilution, reading pure per-spot scatter as a several-percent distance error."""
    rng = np.random.default_rng(0)
    plan = np.repeat(np.linspace(-52.7, 52.7, 19), 400)
    measured = plan + rng.normal(0.0, 8.0, plan.size)

    fit = fit_position_scale(plan, measured)
    assert fit is not None
    assert abs(fit.gain - 1.0) < 0.01

    diluted_slope = float(np.polyfit(measured, plan, 1)[0])
    assert abs(1.0 / diluted_slope - 1.0) > 0.05
    # The scatter is also large enough that no real gain is resolvable here.
    assert fit.gain_significance < 3.0


def test_offset_only_fit_would_be_biased() -> None:
    """Why the fit is joint: a wrong gain leaks into an offset-only correction."""
    plan = np.linspace(-100.0, 100.0, 41)
    measured = _delivered_positions(plan)
    offset_only = _CONFIG_OFFSET - float(np.median(measured - plan))
    fit = fit_position_scale(plan, measured)
    assert fit is not None
    joint_offset = (_CONFIG_OFFSET - fit.bias) / fit.gain
    assert abs(joint_offset - _TRUE_OFFSET) < abs(offset_only - _TRUE_OFFSET)


def test_flat_plan_cannot_identify_a_gain() -> None:
    plan = np.zeros(40)
    assert fit_position_scale(plan, plan + 0.4) is None
    narrow = np.linspace(-MIN_PLAN_SPAN_MM / 4.0, MIN_PLAN_SPAN_MM / 4.0, 40)
    assert fit_position_scale(narrow, narrow) is None


def test_too_few_samples_is_rejected() -> None:
    plan = np.linspace(-50.0, 50.0, 4)
    assert fit_position_scale(plan, plan) is None


def test_weights_are_honoured() -> None:
    plan = np.linspace(-100.0, 100.0, 41)
    corrupted = _delivered_positions(plan)
    corrupted[0] += 50.0
    weights = np.ones_like(plan)
    weights[0] = 0.0
    fit = fit_position_scale(plan, corrupted, weights=weights)
    assert fit is not None
    assert abs(_CONFIG_SDD * fit.gain - _TRUE_SDD) < 1e-6
    assert fit.n_samples == 40


def test_collect_rows_reports_distance_move_and_updates_all_chambers() -> None:
    plan = np.linspace(-100.0, 100.0, 41)
    root = ET.fromstring(_devices_xml())
    pairs, warnings = collect_distance_rows(root, _measured(plan, _delivered_positions(plan)))
    assert len(pairs) == 4
    assert warnings == []
    for _, row in pairs:
        assert abs(row.new_sdd_mm - _TRUE_SDD) < 1e-6
        assert abs(row.new_offset_mm - _TRUE_OFFSET) < 1e-9
        assert abs(row.delta_sdd_mm - (_TRUE_SDD - _CONFIG_SDD)) < 1e-6
        assert abs(row.delta_sdd_percent - (-4.0)) < 1e-6
        assert row.is_large_move


def test_apply_writes_both_elements_together() -> None:
    plan = np.linspace(-100.0, 100.0, 41)
    root = ET.fromstring(_devices_xml())
    result = apply_ic_distances_to_tree(root, _measured(plan, _delivered_positions(plan)))
    assert result.ok
    assert result.distances_updated == 4
    for chamber in root.iter("ion_chamber"):
        assert abs(float(chamber.findtext("source_to_device_distance_mm")) - _TRUE_SDD) < 1e-3
        assert abs(float(chamber.findtext("zero_offset_at_iso_mm")) - _TRUE_OFFSET) < 1e-3


def test_reapplying_a_tuned_config_is_a_no_op() -> None:
    """Tuning must converge: rerun on the tuned config and nothing moves."""
    plan = np.linspace(-100.0, 100.0, 41)
    strips = _delivered_strips(plan)
    root = ET.fromstring(_devices_xml())
    apply_ic_distances_to_tree(root, _measured(plan, _reported(strips, _CONFIG_SDD, _CONFIG_OFFSET)))

    chamber = next(root.iter("ion_chamber"))
    tuned_sdd = float(chamber.findtext("source_to_device_distance_mm"))
    tuned_offset = float(chamber.findtext("zero_offset_at_iso_mm"))
    redelivered = _reported(strips, tuned_sdd, tuned_offset)
    assert float(np.max(np.abs(redelivered - plan))) < 1e-3

    pairs, _ = collect_distance_rows(root, _measured(plan, redelivered))
    assert len(pairs) == 4
    for _, row in pairs:
        assert abs(row.delta_sdd_mm) < 1e-3
        assert abs(row.delta_offset_mm) < 1e-6


def test_implausible_gain_is_refused() -> None:
    plan = np.linspace(-100.0, 100.0, 41)
    wild = 1.0 + 2.0 * MAX_ABS_GAIN_DEVIATION
    root = ET.fromstring(_devices_xml())
    pairs, warnings = collect_distance_rows(root, _measured(plan, plan * wild))
    assert pairs == []
    assert any("too far from 1" in w for w in warnings)


def test_backwards_axis_is_named_not_reported_as_unfittable() -> None:
    """A reversed axis is a sign problem; no positive distance can fix it."""
    plan = np.linspace(-100.0, 100.0, 41)
    fit = fit_position_scale(plan, -plan)
    assert fit is not None and fit.gain < 0.0

    root = ET.fromstring(_devices_xml())
    pairs, warnings = collect_distance_rows(root, _measured(plan, -plan))
    assert pairs == []
    assert any("too far from 1" in w for w in warnings)
    assert not any("not enough spread" in w for w in warnings)


def test_timeslice_data_without_plan_positions_is_refused() -> None:
    plan = np.linspace(-100.0, 100.0, 41)
    energies = np.full(plan.shape, 150.0)
    no_plan = MeasuredPositionErrors(
        by_device={d: (energies, plan * 0.0) for d in ("IC_1_X", "IC_1_Y")},
    )
    root = ET.fromstring(_devices_xml())
    pairs, warnings = collect_distance_rows(root, no_plan)
    assert pairs == []
    assert any("spot" in w for w in warnings)


def test_change_within_fit_uncertainty_is_flagged() -> None:
    rng = np.random.default_rng(1)
    plan = np.repeat(np.linspace(-52.7, 52.7, 19), 400)
    measured = plan + rng.normal(0.0, 8.0, plan.size) + _CONFIG_OFFSET
    root = ET.fromstring(_devices_xml())
    pairs, _ = collect_distance_rows(root, _measured(plan, measured))
    assert pairs
    rows = [row for _, row in pairs]
    assert all(row.is_within_noise for row in rows)
    assert all(abs(row.delta_sdd_mm) < 3.0 * row.sdd_stderr_mm for row in rows)
    warnings = physicality_warnings(rows)
    assert any("within the fit's own uncertainty" in w for w in warnings)
    assert all(w.isascii() for w in warnings)


def test_sigma_recalibration_and_survey_are_flagged() -> None:
    plan = np.linspace(-100.0, 100.0, 41)
    root = ET.fromstring(_devices_xml())
    pairs, _ = collect_distance_rows(root, _measured(plan, _delivered_positions(plan)))
    warnings = physicality_warnings([row for _, row in pairs])
    assert any("Sigma Tuning" in w for w in warnings)
    assert any("surveyed distance" in w for w in warnings)
    assert all(w.isascii() for w in warnings)


def test_merge_drops_plan_positions_when_a_session_lacks_them() -> None:
    plan = np.linspace(-10.0, 10.0, 5)
    energies = np.full(plan.shape, 150.0)
    with_plan = MeasuredPositionErrors(
        by_device={"IC_1_X": (energies, plan * 0.0)},
        plan_by_device={"IC_1_X": plan},
    )
    without_plan = MeasuredPositionErrors(by_device={"IC_1_X": (energies, plan * 0.0)})
    merged = merge_measured_position_errors([with_plan, with_plan])
    assert merged is not None
    assert len(merged.plan_by_device["IC_1_X"]) == 10
    mixed = merge_measured_position_errors([with_plan, without_plan])
    assert mixed is not None
    assert mixed.plan_by_device == {}


def test_outliers_at_the_field_edge_are_rejected_before_fitting() -> None:
    """Least squares has no breakdown resistance: edge outliers pull the gain hardest.

    Twenty one-sided bad spots at one edge, a plausible stuck-readback failure, would
    shift the fitted distance well past its own error bar if they were kept.
    """
    rng = np.random.default_rng(7)
    plan = np.repeat(np.linspace(-125.0, 125.0, 11), 100)
    measured = _reported(_delivered_strips(plan), _CONFIG_SDD, _CONFIG_OFFSET)
    measured += rng.normal(0.0, 0.3, plan.size)

    corrupted = measured.copy()
    edge = np.flatnonzero(plan <= -125.0)[:20]
    corrupted[edge] += 16.0

    clean = fit_position_scale(plan, measured)
    robust = fit_position_scale(plan, corrupted)
    naive = fit_position_scale(plan, corrupted, clip_sigma=float("inf"))
    assert clean is not None and robust is not None and naive is not None

    assert robust.n_rejected == 20
    assert clean.n_rejected == 0
    assert naive.n_rejected == 0

    truth = _CONFIG_SDD * clean.gain
    # Keeping them corrupts the distance; rejecting them recovers the clean answer.
    assert abs(_CONFIG_SDD * naive.gain - truth) > 1.0
    assert abs(_CONFIG_SDD * robust.gain - truth) < 0.1


def test_rejected_spots_still_show_up_in_the_residuals() -> None:
    """A filter that hides the bad data as well as excluding it is worse than none."""
    plan = np.repeat(np.linspace(-125.0, 125.0, 11), 100)
    measured = _reported(_delivered_strips(plan), _CONFIG_SDD, _CONFIG_OFFSET)
    measured += np.random.default_rng(3).normal(0.0, 0.3, plan.size)
    measured[0] += 40.0

    fit = fit_position_scale(plan, measured)
    assert fit is not None
    assert fit.n_rejected >= 1
    assert fit.max_abs_before_mm > 30.0
    assert fit.max_abs_after_mm > 30.0


def test_heavy_rejection_is_reported_as_a_data_problem() -> None:
    plan = np.repeat(np.linspace(-125.0, 125.0, 11), 100)
    measured = _reported(_delivered_strips(plan), _CONFIG_SDD, _CONFIG_OFFSET)
    rng = np.random.default_rng(11)
    measured += rng.normal(0.0, 0.3, plan.size)
    measured[rng.choice(plan.size, 200, replace=False)] += 20.0

    root = ET.fromstring(_devices_xml())
    pairs, _ = collect_distance_rows(root, _measured(plan, measured))
    rows = [row for _, row in pairs]
    assert rows and all(row.rejected_fraction > REJECTED_FRACTION_WARN for row in rows)
    assert any("rejected as outliers" in w for w in physicality_warnings(rows))


def test_exact_fit_has_no_scale_to_clip_against() -> None:
    """Synthetic perfect data has zero MAD; clipping must not eat the whole sample."""
    plan = np.linspace(-100.0, 100.0, 41)
    fit = fit_position_scale(plan, _delivered_positions(plan))
    assert fit is not None
    assert fit.n_rejected == 0
    assert fit.n_samples == plan.size


def test_systematic_is_the_modelled_error_at_the_field_edge() -> None:
    plan = np.linspace(-100.0, 100.0, 41)
    fit = fit_position_scale(plan, _delivered_positions(plan))
    assert fit is not None
    # The fitted line evaluated at the worst edge, which is what the change removes.
    edges = [(fit.gain - 1.0) * p + fit.bias for p in (-100.0, 100.0)]
    assert abs(fit.systematic_removed_mm - max(abs(e) for e in edges)) < 1e-12
    assert fit.systematic_removed_mm > 1.0


def test_real_field_session_improves_rms_even_where_one_outlier_grows() -> None:
    """Session 1242721320: a dropped spot dominates max |err| on X.

    Least squares minimises RMS, and the X systematic has the opposite sign to that
    outlier, so correcting the geometry unmasks it. This pins that the preview keeps
    reporting the honest numbers rather than being "fixed" to hide the rise.
    """
    session = "1242721320"
    devices = _TEST_DATA / session / session / "config" / "map2map" / "devices.xml"
    root = ET.fromstring(devices.read_text(encoding="utf-8"))
    measured = load_measured_position_errors_spot(session, str(_TEST_DATA))
    assert measured is not None
    pairs, _ = collect_distance_rows(root, measured)
    rows = {row.device: row for _, row in pairs}
    assert set(rows) == {"IC_1_X", "IC_1_Y", "IC_2_X", "IC_2_Y"}

    for row in rows.values():
        # Every axis carries a real, highly significant systematic worth ~0.5-1.3 mm.
        assert row.gain_significance > 30.0
        assert row.systematic_removed_mm > 0.4
        assert row.rms_after_mm < row.rms_before_mm

    # X is where the outlier lives: RMS roughly halves while the worst spot grows.
    for device in ("IC_1_X", "IC_2_X"):
        row = rows[device]
        assert row.rms_after_mm < 0.6 * row.rms_before_mm
        assert row.max_abs_before_mm > 10.0
        assert row.max_abs_worsened
        # The rise is bounded by the systematic being removed, not runaway divergence.
        assert row.max_abs_after_mm - row.max_abs_before_mm < 1.5 * row.systematic_removed_mm


def test_distance_formatting_stays_decimal() -> None:
    assert format_distance_mm(1496.737) == "1496.737"
    assert format_distance_mm(1180.0) == "1180"
    assert format_distance_mm(0.0) == "0"
