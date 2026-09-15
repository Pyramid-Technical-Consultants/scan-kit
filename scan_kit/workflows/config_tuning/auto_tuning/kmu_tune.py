"""Scale ion-chamber ``K_MU`` so secondaries agree with the primary.

``K_MU`` is coulombs per monitor unit on each ``<ion_chamber>``
``gain_conversion`` with ``in_units="MU"``. map2map converts
``MU = Q / K_MU``; scan-kit session CSVs already store that converted MU in
``ic*_total_dose_spot``.

The primary chamber is usually the beam terminator, so
``sum(primary MU) ≈ sum(CHARGE_REQ)`` by construction. Matching primary
``K_MU`` to the plan would be circular and would not change dose at
isocenter. This tuner therefore:

- leaves primary ``K_MU`` unchanged by default
- scales every secondary family so its total MU would have matched the
  (optionally rescaled) primary
- optionally rescales the primary first, either to a known MU delivered at
  isocenter or by a percentage of reported MU

Positive percent means more reported MU / more delivered charge for the
same prescription, so ``K_MU`` decreases: ``+2%`` → scale ``1/1.02``.

One scale per IC family (IC1 / IC2 / IC3). HCC and strip devices in that
family keep their relative ``K_MU`` (different collection electrodes);
they all move by the same factor. Session CSVs give one dose series per
IC, not per electrode.

Estimator is the pooled sum ``Σ MU_ic / MU_target`` across selected
sessions. That is the dose-weighted mean of per-spot ratios; averaging
per-session ratios would equally weight a 1 MU map and a 100 MU map.

# ponytail: upstream foil loss between IC1 and IC2 is ~0.1% at therapy
# energies and is left unmodeled. If a room ever needs sub-0.1% relative
# calibration, apply a geometry-dependent transmission factor on this
# ratio rather than inventing per-spot weights.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from scan_kit.common.processing import load_session_raw
from scan_kit.common.schema import (
    C_CHARGE_REQ,
    C_ENERGY,
    resolve_column_name,
    resolve_concept_column,
)

from .sigma_tune import format_sigma_k0

_log = logging.getLogger(__name__)

KmuPrimaryIc = Literal["ic1", "ic2", "ic3"]
KmuPrimaryMode = Literal["unchanged", "known_mu", "percent"]

IC_FAMILIES: tuple[KmuPrimaryIc, ...] = ("ic1", "ic2", "ic3")
DEFAULT_KMU_PRIMARY_IC: KmuPrimaryIc = "ic1"
DEFAULT_KMU_PRIMARY_MODE: KmuPrimaryMode = "unchanged"

MIN_SPOTS = 8
SCALE_WARN_PERCENT = 5.0
PLAN_DISAGREE_WARN_PERCENT = 1.0
KNOWN_MU_TYPO_WARN_PERCENT = 20.0
ENERGY_RESIDUAL_WARN_PERCENT = 2.0

_MU_CANDIDATES: dict[str, tuple[str, ...]] = {
    "ic1": ("ic1_total_dose_spot", "ic1_total_dose"),
    "ic2": ("ic2_total_dose_spot", "ic2_total_dose"),
    "ic3": (
        "r_ic3_total_dose_spot",
        "ic3_total_dose_spot",
        "ic3_total_dose",
        "r_ic3_total_dose",
    ),
}

_DEVICE_ORDER = {
    "IC_1_X": 0,
    "IC_1_Y": 1,
    "IC_1_HCC": 2,
    "IC_2_X": 3,
    "IC_2_Y": 4,
    "IC_2_HCC": 5,
    "IC_3": 6,
}


def is_raw_dose_column(name: str) -> bool:
    """True when *name* is a raw nC dose column, not processed MU."""
    return "_raw" in str(name).strip().lower()


def resolve_spot_dose_mu_column(columns, ic: str) -> str | None:
    """Return the processed MU column for one IC, never a ``*_raw`` nC column.

    Do not use ``C_IC*_TOTAL_DOSE``: that alias list can fall through to
    ``ic*_total_dose_spot_raw`` (nanocoulombs) and would corrupt the scale.
    """
    for candidate in _MU_CANDIDATES.get(ic, ()):
        resolved = resolve_column_name(columns, candidate)
        if resolved is None or is_raw_dose_column(resolved):
            continue
        return resolved
    return None


def ic_family_from_device(name: str) -> KmuPrimaryIc | None:
    """Map a ``devices.xml`` ion-chamber name onto IC1 / IC2 / IC3."""
    upper = str(name).strip().upper()
    if upper.startswith("IC_1"):
        return "ic1"
    if upper.startswith("IC_2"):
        return "ic2"
    if upper.startswith("IC_3"):
        return "ic3"
    return None


def normalize_kmu_primary_ic(value: object) -> KmuPrimaryIc:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in IC_FAMILIES:
        return text  # type: ignore[return-value]
    return DEFAULT_KMU_PRIMARY_IC


def normalize_kmu_primary_mode(value: object) -> KmuPrimaryMode:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in ("unchanged", "known_mu", "percent"):
        return text  # type: ignore[return-value]
    return DEFAULT_KMU_PRIMARY_MODE


def normalize_kmu_known_mu(value: object) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(parsed):
        return float("nan")
    return parsed


def normalize_kmu_percent(value: object) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(parsed):
        return 0.0
    return parsed


@dataclass(frozen=True)
class MeasuredDoseSpots:
    """Aligned per-spot MU for each IC family.

    Missing chambers are NaN of the same length so pairwise masks stay valid
    when sessions are concatenated.
    """

    mu_by_ic: dict[str, np.ndarray]
    energy: np.ndarray | None = None
    charge_req: np.ndarray | None = None

    @property
    def n_spots(self) -> int:
        if not self.mu_by_ic:
            return 0
        return int(next(iter(self.mu_by_ic.values())).size)


@dataclass(frozen=True)
class KmuTunePreviewRow:
    """Proposed ``K_MU`` for one ``ion_chamber`` device."""

    device: str
    family: str
    role: str
    n_spots: int
    measured_mu: float
    plan_mu: float
    vs_primary_pct_before: float
    vs_primary_pct_after: float
    vs_plan_pct: float
    old_kmu: float
    new_kmu: float
    scale: float
    will_write: bool
    max_energy_residual_pct: float

    @property
    def delta_pct(self) -> float:
        if self.old_kmu == 0.0 or not np.isfinite(self.old_kmu):
            return float("nan")
        return 100.0 * (self.new_kmu - self.old_kmu) / self.old_kmu

    @property
    def is_large_scale(self) -> bool:
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            return False
        return abs(self.scale - 1.0) * 100.0 > SCALE_WARN_PERCENT


@dataclass
class KmuTuneResult:
    devices_updated: int = 0
    rows: list[KmuTunePreviewRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    primary_sum_mu: float = float("nan")
    plan_sum_mu: float = float("nan")
    mu_target: float = float("nan")
    primary: str = DEFAULT_KMU_PRIMARY_IC
    mode: str = DEFAULT_KMU_PRIMARY_MODE

    @property
    def ok(self) -> bool:
        return self.devices_updated > 0


@dataclass
class _KmuElement:
    device: str
    family: str
    element: ET.Element
    attr: str
    old_kmu: float


def load_measured_dose_spots(
    session_id: str,
    base_dir: str | Path,
) -> MeasuredDoseSpots | None:
    """Per-spot processed MU for each IC, row-aligned with the input map."""
    input_map, spot_data = load_session_raw(session_id, base_dir=base_dir)
    if input_map is None or spot_data is None:
        return None

    n = min(len(input_map), len(spot_data))
    if n <= 0:
        return None
    input_map = input_map.iloc[:n]
    spot_data = spot_data.iloc[:n]

    mu_by_ic: dict[str, np.ndarray] = {}
    for ic in IC_FAMILIES:
        col = resolve_spot_dose_mu_column(spot_data.columns, ic)
        if col is None:
            mu_by_ic[ic] = np.full(n, np.nan, dtype=float)
            continue
        mu_by_ic[ic] = pd.to_numeric(spot_data[col], errors="coerce").to_numpy(
            dtype=float
        )

    if all(not np.any(np.isfinite(values) & (values > 0)) for values in mu_by_ic.values()):
        _log.debug("Session %s: no processed IC MU columns", session_id)
        return None

    energy: np.ndarray | None = None
    energy_col = resolve_concept_column(input_map.columns, C_ENERGY)
    if energy_col is not None:
        energy = pd.to_numeric(input_map[energy_col], errors="coerce").to_numpy(
            dtype=float
        )

    charge_req: np.ndarray | None = None
    plan_col = resolve_concept_column(input_map.columns, C_CHARGE_REQ)
    if plan_col is not None:
        charge_req = pd.to_numeric(input_map[plan_col], errors="coerce").to_numpy(
            dtype=float
        )

    return MeasuredDoseSpots(mu_by_ic=mu_by_ic, energy=energy, charge_req=charge_req)


def merge_measured_dose_spots(parts: list[MeasuredDoseSpots]) -> MeasuredDoseSpots | None:
    """Concatenate per-spot MU from multiple sessions."""
    usable = [part for part in parts if part.n_spots > 0]
    if not usable:
        return None

    mu_by_ic: dict[str, np.ndarray] = {}
    for ic in IC_FAMILIES:
        chunks = [part.mu_by_ic[ic] for part in usable if ic in part.mu_by_ic]
        if chunks:
            mu_by_ic[ic] = np.concatenate(chunks)

    energy_chunks = [part.energy for part in usable if part.energy is not None]
    energy = np.concatenate(energy_chunks) if len(energy_chunks) == len(usable) else None

    plan_chunks = [part.charge_req for part in usable if part.charge_req is not None]
    charge_req = np.concatenate(plan_chunks) if len(plan_chunks) == len(usable) else None

    if not mu_by_ic:
        return None
    return MeasuredDoseSpots(mu_by_ic=mu_by_ic, energy=energy, charge_req=charge_req)


def load_measured_dose_spots_for_sessions(
    session_ids: list[str],
    base_dir: str | Path,
) -> tuple[MeasuredDoseSpots | None, list[str]]:
    """Load and merge processed spot MU from every resolved *session_ids* entry."""
    warnings: list[str] = []
    parts: list[MeasuredDoseSpots] = []
    for session_id in session_ids:
        sid = str(session_id).strip()
        if not sid:
            continue
        spots = load_measured_dose_spots(sid, base_dir)
        if spots is None:
            warnings.append(f"Could not load dose data for session {sid!r}.")
            continue
        parts.append(spots)

    merged = merge_measured_dose_spots(parts)
    if merged is None:
        if not warnings:
            warnings.append("No session dose data could be loaded.")
        return None, warnings
    return merged, warnings


def _pair_mask(primary: np.ndarray, other: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(primary)
        & np.isfinite(other)
        & (primary > 0.0)
        & (other > 0.0)
    )


def _sum_on_mask(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.sum(values[mask]))


def _pct_diff(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) < 1e-15:
        return float("nan")
    return 100.0 * (value - reference) / reference


def _max_energy_residual_pct(
    mu_ic: np.ndarray,
    mu_pri: np.ndarray,
    energy: np.ndarray | None,
    global_scale: float,
) -> float:
    """Largest per-energy ``Σ MU_ic / Σ MU_pri`` disagreement with *global_scale*."""
    if energy is None or not np.isfinite(global_scale) or abs(global_scale) < 1e-15:
        return float("nan")
    mask = _pair_mask(mu_pri, mu_ic) & np.isfinite(energy)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")
    worst = 0.0
    found = False
    for energy_value in np.unique(energy[mask]):
        layer = mask & (energy == energy_value)
        sum_pri = _sum_on_mask(mu_pri, layer)
        sum_ic = _sum_on_mask(mu_ic, layer)
        if sum_pri <= 0.0 or sum_ic <= 0.0:
            continue
        layer_scale = sum_ic / sum_pri
        residual = abs(layer_scale - global_scale) / abs(global_scale) * 100.0
        if residual > worst:
            worst = residual
        found = True
    return worst if found else float("nan")


def _mu_gain_attr(element: ET.Element) -> str | None:
    if element.get("K_MU") is not None:
        return "K_MU"
    if element.get("k_mu") is not None:
        return "k_mu"
    return None


def read_kmu_elements(root: ET.Element) -> list[_KmuElement]:
    """Every ion-chamber MU ``gain_conversion`` this tuner may rewrite."""
    found: list[_KmuElement] = []
    for chamber in root.iter("ion_chamber"):
        device_el = chamber.find("device")
        if device_el is None:
            continue
        name = (device_el.get("name") or "").strip()
        family = ic_family_from_device(name)
        if family is None:
            continue
        for conv in chamber.iter("gain_conversion"):
            if (conv.get("in_units") or "").upper() != "MU":
                continue
            attr = _mu_gain_attr(conv)
            if attr is None:
                continue
            try:
                old = float((conv.get(attr) or "").strip())
            except (TypeError, ValueError):
                continue
            if not np.isfinite(old) or old == 0.0:
                continue
            found.append(
                _KmuElement(
                    device=name,
                    family=family,
                    element=conv,
                    attr=attr,
                    old_kmu=old,
                )
            )
            break
    found.sort(key=lambda item: (_DEVICE_ORDER.get(item.device, 99), item.device))
    return found


def compute_family_totals(
    measured: MeasuredDoseSpots,
    primary: KmuPrimaryIc,
) -> dict[str, tuple[int, float, float, float]]:
    """Per-IC ``(n_spots, sum_mu, sum_primary, sum_plan)`` on the pairwise mask."""
    mu_pri = measured.mu_by_ic.get(primary)
    if mu_pri is None:
        return {}
    totals: dict[str, tuple[int, float, float, float]] = {}
    for ic, mu_ic in measured.mu_by_ic.items():
        if mu_ic.shape != mu_pri.shape:
            continue
        mask = _pair_mask(mu_pri, mu_ic)
        n_spots = int(np.count_nonzero(mask))
        sum_ic = _sum_on_mask(mu_ic, mask)
        sum_pri = _sum_on_mask(mu_pri, mask)
        sum_plan = float("nan")
        if measured.charge_req is not None and measured.charge_req.shape == mu_ic.shape:
            plan_mask = mask & np.isfinite(measured.charge_req) & (measured.charge_req > 0.0)
            sum_plan = _sum_on_mask(measured.charge_req, plan_mask)
        totals[ic] = (n_spots, sum_ic, sum_pri, sum_plan)
    return totals


def compute_mu_target(
    totals: dict[str, tuple[int, float, float, float]],
    primary: KmuPrimaryIc,
    mode: KmuPrimaryMode,
    known_mu: float,
    percent: float,
) -> tuple[float | None, list[str]]:
    """Return ``MU_target`` and any mode-specific warnings/errors."""
    warnings: list[str] = []
    primary_totals = totals.get(primary)
    if primary_totals is None:
        return None, ["No processed MU for the primary IC."]
    _, _, sum_pri, _ = primary_totals
    if not np.isfinite(sum_pri) or sum_pri <= 0.0:
        return None, ["Primary IC has no positive MU in the selected sessions."]

    if mode == "known_mu":
        if not np.isfinite(known_mu) or known_mu <= 0.0:
            warnings.append("Enter a known MU at isocenter greater than 0.")
            return None, warnings
        mismatch = abs(_pct_diff(known_mu, sum_pri))
        if mismatch > KNOWN_MU_TYPO_WARN_PERCENT:
            warnings.append(
                f"Known MU {known_mu:g} is {mismatch:.1f}% from primary "
                f"{sum_pri:.4g} MU. Check the isocenter reading."
            )
        return known_mu, warnings

    if mode == "percent":
        if percent <= -100.0:
            warnings.append("Percent adjustment must be greater than -100.")
            return None, warnings
        return sum_pri * (1.0 + percent / 100.0), warnings

    return sum_pri, warnings


def _family_scale(
    sum_ic: float,
    mu_target: float,
) -> float:
    if not np.isfinite(sum_ic) or not np.isfinite(mu_target) or mu_target <= 0.0:
        return float("nan")
    return sum_ic / mu_target


def collect_kmu_updates(
    root: ET.Element,
    measured: MeasuredDoseSpots,
    *,
    primary: KmuPrimaryIc = DEFAULT_KMU_PRIMARY_IC,
    mode: KmuPrimaryMode = DEFAULT_KMU_PRIMARY_MODE,
    known_mu: float = float("nan"),
    percent: float = 0.0,
) -> tuple[list[tuple[_KmuElement, KmuTunePreviewRow]], list[str], float, float, float]:
    """Build preview rows without mutating *root*.

    Returns ``(pairs, warnings, primary_sum_mu, plan_sum_mu, mu_target)``.
    """
    warnings: list[str] = []
    totals = compute_family_totals(measured, primary)
    primary_totals = totals.get(primary)
    if primary_totals is None:
        return [], ["No processed MU for the primary IC."], float("nan"), float("nan"), float("nan")

    n_pri, sum_pri_all, _, plan_pri = primary_totals
    mu_target, mode_warnings = compute_mu_target(
        totals, primary, mode, known_mu, percent
    )
    warnings.extend(mode_warnings)
    if mu_target is None:
        return [], warnings, sum_pri_all, plan_pri, float("nan")

    if n_pri < MIN_SPOTS:
        warnings.append(
            f"Primary {primary.upper()} has {n_pri} valid spots; need {MIN_SPOTS}."
        )

    if np.isfinite(plan_pri) and plan_pri > 0.0:
        plan_err = abs(_pct_diff(sum_pri_all, plan_pri))
        if plan_err > PLAN_DISAGREE_WARN_PERCENT:
            warnings.append(
                f"Primary reports {sum_pri_all:.4g} MU vs plan {plan_pri:.4g} MU "
                f"({plan_err:.2f}%). Aborted spots, or the primary is not the "
                f"terminator; secondary matching still uses delivered spots."
            )

    mu_pri = measured.mu_by_ic.get(primary)
    scales: dict[str, float] = {}
    write_ok: dict[str, bool] = {}
    energy_residual: dict[str, float] = {}

    for ic, (n_spots, sum_ic, sum_pri, _sum_plan) in totals.items():
        scale = _family_scale(sum_ic, mu_target)
        scales[ic] = scale
        residual = float("nan")
        if mu_pri is not None and ic in measured.mu_by_ic and np.isfinite(sum_pri) and sum_pri > 0.0:
            residual = _max_energy_residual_pct(
                measured.mu_by_ic[ic],
                mu_pri,
                measured.energy,
                sum_ic / sum_pri,
            )
        energy_residual[ic] = residual

        is_primary = ic == primary
        skip_primary = is_primary and mode == "unchanged"
        skip_write = False
        if n_spots < MIN_SPOTS:
            warnings.append(
                f"{ic.upper()}: {n_spots} valid spots; need {MIN_SPOTS} to write K_MU."
            )
            skip_write = True
        elif not np.isfinite(scale) or scale <= 0.0:
            warnings.append(f"{ic.upper()}: could not compute a K_MU scale.")
            skip_write = True
        elif abs(scale - 1.0) * 100.0 > SCALE_WARN_PERCENT and not skip_primary:
            warnings.append(
                f"{ic.upper()}: K_MU scale {scale:.4f} "
                f"({(scale - 1.0) * 100.0:+.2f}%). Commissioning often needs a "
                f"move this large; review before applying."
            )

        if (
            not is_primary
            and np.isfinite(residual)
            and residual > ENERGY_RESIDUAL_WARN_PERCENT
        ):
            warnings.append(
                f"{ic.upper()}: per-energy MU ratio vs primary wanders "
                f"{residual:.1f}% from the global scale. A single K_MU cannot "
                f"make the chambers agree at every layer."
            )

        write_ok[ic] = (not skip_primary) and (not skip_write)

    primary_scale = scales.get(primary, 1.0)
    pairs: list[tuple[_KmuElement, KmuTunePreviewRow]] = []
    for element in read_kmu_elements(root):
        family_totals = totals.get(element.family)
        if family_totals is None:
            warnings.append(f"No session MU for {element.device} ({element.family}).")
            continue
        n_spots, sum_ic, sum_pri, sum_plan = family_totals
        scale = scales.get(element.family, float("nan"))
        skip_primary = element.family == primary and mode == "unchanged"
        if skip_primary or not np.isfinite(scale):
            new_kmu = element.old_kmu
            applied_scale = 1.0 if skip_primary else scale
        else:
            new_kmu = element.old_kmu * scale
            applied_scale = scale

        vs_pri_before = 0.0 if element.family == primary else _pct_diff(sum_ic, sum_pri)
        if skip_primary:
            vs_pri_after = 0.0
        elif np.isfinite(scale) and np.isfinite(primary_scale) and primary_scale != 0.0:
            vs_pri_after = _pct_diff(sum_ic / scale, sum_pri / primary_scale)
        else:
            vs_pri_after = float("nan")

        row = KmuTunePreviewRow(
            device=element.device,
            family=element.family,
            role="primary" if element.family == primary else "secondary",
            n_spots=n_spots,
            measured_mu=sum_ic,
            plan_mu=sum_plan,
            vs_primary_pct_before=vs_pri_before,
            vs_primary_pct_after=vs_pri_after,
            vs_plan_pct=_pct_diff(sum_ic, sum_plan),
            old_kmu=element.old_kmu,
            new_kmu=new_kmu,
            scale=1.0 if skip_primary else applied_scale,
            will_write=bool(write_ok.get(element.family, False)),
            max_energy_residual_pct=energy_residual.get(element.family, float("nan")),
        )
        pairs.append((element, row))

    if not pairs and not warnings:
        warnings.append("No ion chamber MU gain_conversion matched the session data.")
    return pairs, warnings, sum_pri_all, plan_pri, mu_target


def compute_kmu_tune_preview(
    root: ET.Element,
    session_ids: list[str],
    base_dir: str,
    *,
    primary: KmuPrimaryIc = DEFAULT_KMU_PRIMARY_IC,
    mode: KmuPrimaryMode = DEFAULT_KMU_PRIMARY_MODE,
    known_mu: float = float("nan"),
    percent: float = 0.0,
) -> tuple[list[KmuTunePreviewRow], list[str]]:
    """Return proposed ``K_MU`` values for every matching chamber in *root*."""
    measured, load_warnings = load_measured_dose_spots_for_sessions(session_ids, base_dir)
    if measured is None:
        return [], load_warnings
    pairs, warnings, _pri, _plan, _target = collect_kmu_updates(
        root,
        measured,
        primary=primary,
        mode=mode,
        known_mu=known_mu,
        percent=percent,
    )
    return [row for _, row in pairs], load_warnings + warnings


def apply_kmu_to_tree(
    root: ET.Element,
    measured: MeasuredDoseSpots,
    *,
    primary: KmuPrimaryIc = DEFAULT_KMU_PRIMARY_IC,
    mode: KmuPrimaryMode = DEFAULT_KMU_PRIMARY_MODE,
    known_mu: float = float("nan"),
    percent: float = 0.0,
) -> KmuTuneResult:
    """Write scaled ``K_MU`` attributes for families that passed the safeguards."""
    pairs, warnings, primary_sum, plan_sum, mu_target = collect_kmu_updates(
        root,
        measured,
        primary=primary,
        mode=mode,
        known_mu=known_mu,
        percent=percent,
    )
    written = 0
    for element, row in pairs:
        if not row.will_write:
            continue
        element.element.set(element.attr, format_sigma_k0(row.new_kmu))
        written += 1
    return KmuTuneResult(
        devices_updated=written,
        rows=[row for _, row in pairs],
        warnings=warnings,
        primary_sum_mu=primary_sum,
        plan_sum_mu=plan_sum,
        mu_target=mu_target,
        primary=primary,
        mode=mode,
    )


def tune_kmu_from_sessions(
    root: ET.Element,
    session_ids: list[str],
    base_dir: str,
    *,
    primary: KmuPrimaryIc = DEFAULT_KMU_PRIMARY_IC,
    mode: KmuPrimaryMode = DEFAULT_KMU_PRIMARY_MODE,
    known_mu: float = float("nan"),
    percent: float = 0.0,
) -> KmuTuneResult:
    """Load session MU and rewrite ion-chamber ``K_MU`` in *root*."""
    measured, load_warnings = load_measured_dose_spots_for_sessions(session_ids, base_dir)
    if measured is None:
        return KmuTuneResult(warnings=load_warnings, primary=primary, mode=mode)
    result = apply_kmu_to_tree(
        root,
        measured,
        primary=primary,
        mode=mode,
        known_mu=known_mu,
        percent=percent,
    )
    if load_warnings:
        result.warnings = load_warnings + result.warnings
    return result
