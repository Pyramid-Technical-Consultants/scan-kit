"""Fit IC ``source_to_device_distance_mm`` and ``zero_offset_at_iso_mm`` jointly.

Assumes delivery at isocenter is correct and the chambers are the thing that is
mis-scaled, so each chamber is asked to reproduce the plan it was given.

map2map converts a strip reading to millimetres at isocenter as

    iso = mag * strip_to_mm * (strip - center) + zero_offset_at_iso_mm

with ``mag = source_to_isocenter_distance / source_to_device_distance_mm`` (see
:mod:`scan_kit.common.devices_xml`).  ``source_to_isocenter_distance`` is global and
``strip_to_mm`` is the physical strip pitch, so the only legal per-chamber scale knob
is ``source_to_device_distance_mm``.

Writing ``u`` for the signed strip offset in chamber millimetres, the delivered
positions satisfy ``iso_meas = k_old * u + off_old``.  That is exactly invertible, so
the fit needs no raw strips, only the processed positions and the plan.

**Regression direction matters.** The plan position is exact — it is the commanded
value — while the measurement carries chamber noise, so the noisy variable has to be
the dependent one:

    iso_meas ≈ gain * plan + bias   ⟹   sdd_new = sdd_old * gain
                                        off_new = (off_old - bias) / gain

Regressing the other way (plan on measurement) targets the same quantity but is
attenuated by ``var(plan) / (var(plan) + var(noise))``, which turns per-spot scatter
into a fake scale error: a real session with ~8 mm scatter over a 105 mm field reads
as a 6.7% distance error that way, against 0.5% fitted correctly.

Gain and offset are fitted **together** because an offset fitted against a wrong gain
absorbs part of the scale error and comes out biased.  Fixing ``gain = 1`` recovers the
offset-only model in :mod:`position_offset_tune`.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from scan_kit.common.devices_xml import IC_SIGMA_DEVICES
from scan_kit.common.session_position import (
    DEFAULT_POSITION_DATA_SOURCE,
    MeasuredPositionErrors,
    PositionDataSource,
    load_measured_position_errors_for_sessions,
)

# A slope needs lever arm: a plan that never leaves isocenter cannot tell a scale
# error from an offset. Slope uncertainty goes as position noise over span, so with
# sub-millimetre noise this floor keeps the implied distance uncertainty near 1%.
MIN_PLAN_SPAN_MM = 20.0

# Two free parameters need more than two spots before the fit means anything.
MIN_FIT_SAMPLES = 8

# source_to_device_distance_mm is a surveyed distance. A fitted move this large means
# the assumption broke somewhere else (magnet calibration, strip pitch), not that the
# chamber physically moved.
SDD_WARN_PERCENT = 1.0

# Beyond this the fit is not a calibration refinement, it is a different geometry.
MAX_ABS_GAIN_DEVIATION = 0.25

# Standard errors below which a fitted gain is indistinguishable from "no change".
# Scatter alone will fit a small gain on any real session; moving a surveyed distance
# needs the gain to stand clear of its own uncertainty.
MIN_GAIN_SIGNIFICANCE = 3.0

# Robust sigmas past which a spot is treated as bad data rather than a tail. Wide
# enough that a genuine Gaussian tail survives even at ten thousand spots, so anything
# rejected really is a dropped or mis-associated spot.
OUTLIER_CLIP_SIGMA = 5.0

# Above this rejected fraction the session has a data problem worth reporting, not a
# few stray spots.
REJECTED_FRACTION_WARN = 0.01

_DEVICE_ORDER = {name: index for index, name in enumerate(IC_SIGMA_DEVICES)}


def format_distance_mm(value: float) -> str:
    """Plain decimal millimetres, the way ``devices.xml`` stores source distances."""
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text or "0"


@dataclass(frozen=True)
class IcDistanceTunePreviewRow:
    """Proposed distance and offset for one ion chamber."""

    device: str
    n_samples: int
    n_rejected: int
    plan_span_mm: float
    gain: float
    gain_stderr: float
    old_sdd_mm: float
    new_sdd_mm: float
    old_offset_mm: float
    new_offset_mm: float
    systematic_removed_mm: float
    rms_before_mm: float
    rms_after_mm: float
    max_abs_before_mm: float
    max_abs_after_mm: float

    @property
    def max_abs_worsened(self) -> bool:
        """True when the worst single spot reads further out after the change.

        Least squares minimises RMS, not the worst sample, and a systematic of opposite
        sign partly masks an outlier — removing it lets the outlier show its true size.
        Expected on axes with a dropped spot; judge the change on RMS and systematic.
        """
        return self.max_abs_after_mm > self.max_abs_before_mm

    @property
    def delta_sdd_mm(self) -> float:
        return self.new_sdd_mm - self.old_sdd_mm

    @property
    def sdd_stderr_mm(self) -> float:
        """One-sigma uncertainty on the distance, propagated from the gain fit."""
        return self.old_sdd_mm * self.gain_stderr

    @property
    def gain_significance(self) -> float:
        if self.gain_stderr <= 0.0:
            return float("inf")
        return abs(self.gain - 1.0) / self.gain_stderr

    @property
    def is_within_noise(self) -> bool:
        return self.gain_significance < MIN_GAIN_SIGNIFICANCE

    @property
    def rejected_fraction(self) -> float:
        total = self.n_samples + self.n_rejected
        return self.n_rejected / total if total else 0.0

    @property
    def delta_sdd_percent(self) -> float:
        if self.old_sdd_mm == 0.0:
            return float("nan")
        return 100.0 * self.delta_sdd_mm / self.old_sdd_mm

    @property
    def delta_offset_mm(self) -> float:
        return self.new_offset_mm - self.old_offset_mm

    @property
    def is_large_move(self) -> bool:
        return abs(self.delta_sdd_percent) > SDD_WARN_PERCENT


@dataclass
class IcDistanceTuneResult:
    distances_updated: int = 0
    rows: list[IcDistanceTunePreviewRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.distances_updated > 0


@dataclass
class _ChamberElements:
    """The two elements one chamber's fit writes to."""

    sdd_element: ET.Element
    sdd_mm: float
    offset_element: ET.Element
    offset_mm: float


def read_chamber_elements(
    root: ET.Element,
    *,
    devices: tuple[str, ...] = IC_SIGMA_DEVICES,
) -> dict[str, _ChamberElements]:
    """Find the distance and offset elements this tuner rewrites, per device."""
    device_set = set(devices)
    found: dict[str, _ChamberElements] = {}
    for chamber in root.iter("ion_chamber"):
        device_el = chamber.find("device")
        if device_el is None:
            continue
        name = device_el.get("name")
        if not name or name not in device_set:
            continue
        sdd_el = chamber.find("source_to_device_distance_mm")
        offset_el = chamber.find("zero_offset_at_iso_mm")
        if sdd_el is None or offset_el is None:
            continue
        try:
            sdd = float((sdd_el.text or "").strip())
            offset = float((offset_el.text or "").strip())
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(sdd) and np.isfinite(offset) and sdd > 0.0):
            continue
        found[name] = _ChamberElements(
            sdd_element=sdd_el,
            sdd_mm=sdd,
            offset_element=offset_el,
            offset_mm=offset,
        )
    return found


@dataclass(frozen=True)
class PositionScaleFit:
    """Weighted least-squares fit of measured position against commanded position.

    ``n_samples`` counts the spots the line was fitted to, after outlier rejection.
    The residual statistics deliberately cover *every* finite sample including the
    rejected ones, so a bad spot stays visible in ``max_abs_*`` instead of being
    quietly dropped from the report as well as the fit.
    """

    gain: float
    bias: float
    gain_stderr: float
    n_samples: int
    n_rejected: int
    plan_span_mm: float
    systematic_removed_mm: float
    rms_before_mm: float
    rms_after_mm: float
    max_abs_before_mm: float
    max_abs_after_mm: float

    @property
    def gain_significance(self) -> float:
        """How many standard errors the gain sits away from "no change"."""
        if self.gain_stderr <= 0.0:
            return float("inf")
        return abs(self.gain - 1.0) / self.gain_stderr

    @property
    def rejected_fraction(self) -> float:
        total = self.n_samples + self.n_rejected
        return self.n_rejected / total if total else 0.0


@dataclass(frozen=True)
class _LineFit:
    gain: float
    bias: float
    gain_stderr: float
    s_pp: float


def _weighted_line_fit(
    plan: np.ndarray,
    measured: np.ndarray,
    sample_weights: np.ndarray,
) -> _LineFit | None:
    """Closed-form weighted least squares of ``measured`` on ``plan``."""
    total_weight = float(np.sum(sample_weights))
    plan_mean = float(np.sum(sample_weights * plan) / total_weight)
    measured_mean = float(np.sum(sample_weights * measured) / total_weight)
    centered_plan = plan - plan_mean
    s_pp = float(np.sum(sample_weights * centered_plan**2))
    if s_pp <= 0.0 or plan.size < 3:
        return None
    gain = float(
        np.sum(sample_weights * centered_plan * (measured - measured_mean)) / s_pp
    )
    bias = measured_mean - gain * plan_mean
    # A negative gain is a real fit result (the axis reads backwards), so pass it on and
    # let the caller reject it by name rather than reporting it as unfittable data.
    if not (np.isfinite(gain) and np.isfinite(bias)) or abs(gain) < 1e-9:
        return None
    residual = measured - (gain * plan + bias)
    residual_variance = float(
        np.sum(sample_weights * residual**2) / (plan.size - 2)
    )
    return _LineFit(
        gain=gain,
        bias=bias,
        gain_stderr=float(np.sqrt(residual_variance / s_pp)),
        s_pp=s_pp,
    )


def _outlier_mask(residual: np.ndarray, clip_sigma: float) -> np.ndarray | None:
    """Keep-mask rejecting residuals beyond *clip_sigma* robust sigmas.

    Scale comes from the MAD, not the standard deviation: a handful of large outliers
    inflate the standard deviation enough to hide behind it, which is exactly the case
    this needs to catch. Returns ``None`` when there is nothing to reject.
    """
    centre = float(np.median(residual))
    mad = float(np.median(np.abs(residual - centre)))
    if mad <= 0.0:
        return None  # an exact fit has no scale to clip against
    sigma = 1.4826 * mad  # MAD to Gaussian-equivalent sigma
    keep = np.abs(residual - centre) <= clip_sigma * sigma
    return keep if not bool(np.all(keep)) else None


def fit_position_scale(
    plan_mm: np.ndarray,
    measured_mm: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    clip_sigma: float = OUTLIER_CLIP_SIGMA,
) -> PositionScaleFit | None:
    """Fit ``measured ≈ gain * plan + bias``, then express it as a geometry change.

    Rejects gross outliers once before refitting. Least squares has no breakdown
    resistance of its own: a bad spot's pull on the gain is proportional to its
    residual *and* its distance from the field centre, so a dropped spot at the field
    edge is the worst case, and the ones seen in practice are one-sided rather than
    cancelling.

    Returns ``None`` when the samples cannot identify a gain: too few points, or a plan
    with no position span to provide lever arm.
    """
    plan = np.asarray(plan_mm, dtype=float)
    measured = np.asarray(measured_mm, dtype=float)
    if plan.shape != measured.shape:
        return None

    finite = np.isfinite(plan) & np.isfinite(measured)
    w = None if weights is None else np.asarray(weights, dtype=float)
    if w is not None and w.shape == plan.shape:
        finite = finite & np.isfinite(w) & (w > 0.0)
    else:
        w = None

    if int(np.count_nonzero(finite)) < MIN_FIT_SAMPLES:
        return None

    plan = plan[finite]
    measured = measured[finite]
    span = float(np.max(plan) - np.min(plan))
    if span < MIN_PLAN_SPAN_MM:
        return None
    sample_weights = np.ones_like(plan) if w is None else w[finite]

    fit = _weighted_line_fit(plan, measured, sample_weights)
    if fit is None:
        return None

    # ponytail: one rejection pass, not iterated reweighting. Enough for the isolated
    # dropped spots seen in real sessions; if a session ever needs several passes to
    # settle, that is a data problem worth surfacing, not one to iterate away. Upgrade
    # path is Huber IRLS over this same closed form.
    n_rejected = 0
    keep = _outlier_mask(measured - (fit.gain * plan + fit.bias), clip_sigma)
    if keep is not None:
        kept_span = float(np.max(plan[keep]) - np.min(plan[keep]))
        if (
            int(np.count_nonzero(keep)) >= MIN_FIT_SAMPLES
            and kept_span >= MIN_PLAN_SPAN_MM
        ):
            refit = _weighted_line_fit(plan[keep], measured[keep], sample_weights[keep])
            if refit is not None:
                fit = refit
                n_rejected = int(np.count_nonzero(~keep))

    gain, bias = fit.gain, fit.bias

    # Residuals in delivered terms: how far the reconverted measurement lands from the
    # plan, before and after the proposed change. Reconversion inverts the fitted line,
    # so the "after" residual is the fit residual rescaled by the gain.
    before = measured - plan
    after = (measured - bias) / gain - plan

    # The fitted line *is* the position-dependent error, so evaluating it at the field
    # edges gives the error the change removes, free of the per-spot scatter it cannot
    # touch. A linear function peaks at an endpoint of the interval.
    edges = [
        (gain - 1.0) * float(np.min(plan)) + bias,
        (gain - 1.0) * float(np.max(plan)) + bias,
    ]
    return PositionScaleFit(
        gain=gain,
        bias=bias,
        gain_stderr=fit.gain_stderr,
        n_samples=int(plan.size) - n_rejected,
        n_rejected=n_rejected,
        plan_span_mm=span,
        systematic_removed_mm=max(abs(edges[0]), abs(edges[1])),
        rms_before_mm=float(np.sqrt(np.mean(before**2))),
        rms_after_mm=float(np.sqrt(np.mean(after**2))),
        max_abs_before_mm=float(np.max(np.abs(before))),
        max_abs_after_mm=float(np.max(np.abs(after))),
    )


def collect_distance_rows(
    root: ET.Element,
    measured: MeasuredPositionErrors,
    *,
    devices: tuple[str, ...] = IC_SIGMA_DEVICES,
) -> tuple[list[tuple[_ChamberElements, IcDistanceTunePreviewRow]], list[str]]:
    """Fit every chamber without mutating *root*."""
    rows: list[tuple[_ChamberElements, IcDistanceTunePreviewRow]] = []
    warnings: list[str] = []

    if not measured.plan_by_device:
        warnings.append(
            "Fitting IC distances needs the plan position behind each spot, which only "
            "spot data carries. Switch the data source to spot."
        )
        return rows, warnings

    elements = read_chamber_elements(root, devices=devices)
    for device in devices:
        chamber = elements.get(device)
        samples = measured.by_device.get(device)
        plan = measured.plan_by_device.get(device)
        if chamber is None:
            warnings.append(
                f"No source_to_device_distance_mm and zero_offset_at_iso_mm pair "
                f"for {device}."
            )
            continue
        if samples is None or plan is None:
            warnings.append(f"No session position data for {device}.")
            continue

        _, errors = samples
        fit = fit_position_scale(plan, plan + errors, weights=measured.weights)
        if fit is None:
            warnings.append(
                f"{device}: not enough spread to fit a distance "
                f"(needs {MIN_FIT_SAMPLES}+ spots spanning {MIN_PLAN_SPAN_MM:g} mm)."
            )
            continue
        if abs(fit.gain - 1.0) > MAX_ABS_GAIN_DEVIATION:
            warnings.append(
                f"{device}: fitted gain {fit.gain:.4f} is too far from 1 to be a "
                f"distance calibration; left unchanged."
            )
            continue

        row = IcDistanceTunePreviewRow(
            device=device,
            n_samples=fit.n_samples,
            n_rejected=fit.n_rejected,
            plan_span_mm=fit.plan_span_mm,
            gain=fit.gain,
            gain_stderr=fit.gain_stderr,
            old_sdd_mm=chamber.sdd_mm,
            new_sdd_mm=chamber.sdd_mm * fit.gain,
            old_offset_mm=chamber.offset_mm,
            new_offset_mm=(chamber.offset_mm - fit.bias) / fit.gain,
            systematic_removed_mm=fit.systematic_removed_mm,
            rms_before_mm=fit.rms_before_mm,
            rms_after_mm=fit.rms_after_mm,
            max_abs_before_mm=fit.max_abs_before_mm,
            max_abs_after_mm=fit.max_abs_after_mm,
        )
        rows.append((chamber, row))

    if not rows and not warnings:
        warnings.append("No ion chamber matched the session data.")
    rows.sort(key=lambda pair: _DEVICE_ORDER.get(pair[1].device, 99))
    return rows, warnings


def physicality_warnings(rows: list[IcDistanceTunePreviewRow]) -> list[str]:
    """Flag moves that are not believable or not supported, and the sigma coupling."""
    warnings: list[str] = []
    for row in rows:
        if row.rejected_fraction > REJECTED_FRACTION_WARN:
            warnings.append(
                f"{row.device}: {row.n_rejected} of "
                f"{row.n_samples + row.n_rejected} spots "
                f"({100 * row.rejected_fraction:.1f}%) were rejected as outliers before "
                f"fitting. That is more than stray bad spots, so check the session data."
            )
        if row.is_within_noise:
            warnings.append(
                f"{row.device}: distance change {row.delta_sdd_mm:+.2f} mm is within the "
                f"fit's own uncertainty (+/-{row.sdd_stderr_mm:.2f} mm, "
                f"{row.gain_significance:.1f} sigma). Per-spot scatter alone can fit "
                f"this, so leave the surveyed distance alone unless a wider field agrees."
            )
        if row.is_large_move:
            warnings.append(
                f"{row.device}: source_to_device_distance_mm moves "
                f"{row.delta_sdd_mm:+.2f} mm ({row.delta_sdd_percent:+.2f}%). This is a "
                f"surveyed distance, so check the magnet calibration and strip pitch "
                f"before trusting a move this large."
            )
    if any(abs(row.delta_sdd_mm) > 0.0 for row in rows):
        warnings.append(
            "Changing source_to_device_distance_mm rescales isocenter sigma by the same "
            "factor. Re-run Sigma Tuning afterwards."
        )
    return warnings


def compute_ic_distance_tune_preview(
    root: ET.Element,
    session_ids: list[str],
    base_dir: str,
    *,
    data_source: PositionDataSource = DEFAULT_POSITION_DATA_SOURCE,
) -> tuple[list[IcDistanceTunePreviewRow], list[str]]:
    """Return proposed distances and offsets for every chamber in *root*."""
    measured, load_warnings = load_measured_position_errors_for_sessions(
        session_ids,
        base_dir,
        data_source=data_source,
    )
    if measured is None:
        return [], load_warnings
    pairs, warnings = collect_distance_rows(root, measured)
    rows = [row for _, row in pairs]
    return rows, load_warnings + warnings + physicality_warnings(rows)


def apply_ic_distances_to_tree(
    root: ET.Element,
    measured: MeasuredPositionErrors,
    *,
    devices: tuple[str, ...] = IC_SIGMA_DEVICES,
) -> IcDistanceTuneResult:
    """Write fitted ``source_to_device_distance_mm`` and ``zero_offset_at_iso_mm``.

    Both values come from the same fit, so they are written together; applying one
    without the other leaves the chamber worse off than before.
    """
    pairs, warnings = collect_distance_rows(root, measured, devices=devices)
    for chamber, row in pairs:
        chamber.sdd_element.text = format_distance_mm(row.new_sdd_mm)
        chamber.offset_element.text = format_distance_mm(row.new_offset_mm)
    rows = [row for _, row in pairs]
    return IcDistanceTuneResult(
        distances_updated=len(pairs),
        rows=rows,
        warnings=warnings + physicality_warnings(rows),
    )


def tune_ic_distances_from_sessions(
    root: ET.Element,
    session_ids: list[str],
    base_dir: str,
    *,
    data_source: PositionDataSource = DEFAULT_POSITION_DATA_SOURCE,
) -> IcDistanceTuneResult:
    """Load session positions and rewrite IC distances and offsets in *root*."""
    measured, load_warnings = load_measured_position_errors_for_sessions(
        session_ids,
        base_dir,
        data_source=data_source,
    )
    if measured is None:
        return IcDistanceTuneResult(warnings=load_warnings)
    result = apply_ic_distances_to_tree(root, measured)
    if load_warnings:
        result.warnings = load_warnings + result.warnings
    return result
