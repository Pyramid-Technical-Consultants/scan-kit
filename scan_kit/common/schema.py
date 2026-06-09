"""Column schema helpers and concept-to-column mappings.

This module centralizes how scan-kit resolves logical measurement concepts
to physical CSV column names across schema versions (and G2/G3 variants).

Unit conventions
----------------
After canonicalization all IC current columns are in **nanoamperes (nA)**.

* **G3** stores IC current natively in nA in columns like
  ``ic1_primary_channel``, ``ic2_primary_channel``, ``ic3_current_A/B/C/D``.
  No conversion is needed.
* **G2** stores IC current as **charge in coulombs** accumulated over a 1 ms
  timeslice in columns like ``r_ic1_current_dose``, ``r_ic2_current_dose``.
  These are converted to nA during canonicalization:
  ``nA = coulombs / 0.001 s × 1e9 = coulombs × 1e12``.

The ``_COLUMN_SCALE_FACTORS`` dict below lists every G2 column that needs
scaling.  ``canonicalize_dataframe_columns`` applies these factors
automatically whenever it renames a column.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

POSITION_KEY_G2_RAW = "spot_raw"
POSITION_KEY_G3_RAW = "spot_position_raw"
POSITION_KEY_G2 = "spot"
POSITION_KEY_G3 = "spot_position"

# Concept identifiers used throughout processing/view code.
C_ENERGY = "energy"
C_LAYER_ID = "layer_id"
C_SPOT_NO = "spot_no"
C_TIMESTAMP = "timestamp"
C_TIME_S = "time_s"
C_TIME_NS = "time_ns"

# IC current concepts — after canonicalization these are always in nA.
# G3 columns (ic1_primary_channel, etc.) are natively nA.
# G2 columns (r_ic1_current_dose, etc.) are coulombs per 1 ms timeslice;
# they are scaled to nA by _COLUMN_SCALE_FACTORS during canonicalization.
C_IC1_CURRENT = "ic1_current"
C_IC2_CURRENT = "ic2_current"
C_IC3_CURRENT_A = "ic3_current_a"  # G3 only (quad IC3)
C_IC3_CURRENT_B = "ic3_current_b"  # G3 only
C_IC3_CURRENT_C = "ic3_current_c"  # G3 only
C_IC3_CURRENT_D = "ic3_current_d"  # G3 only
C_IC1_STRIP_SUM = "ic1_strip_sum"  # G3 IC1 strip-sum ADC channel
C_IC2_STRIP_SUM = "ic2_strip_sum"  # G3 IC2 strip-sum ADC channel
C_BEAM_CURRENT = "beam_current"
C_IC1_TOTAL_DOSE = "ic1_total_dose"
C_IC2_TOTAL_DOSE = "ic2_total_dose"
C_IC3_TOTAL_DOSE = "ic3_total_dose"
# Timeslice scan-total dose: monotonically accumulating dose across the
# entire scan (does not reset per spot/layer).  Differentiating this column
# yields the IC current.
C_IC1_SCAN_TOTAL_DOSE = "ic1_scan_total_dose"
C_IC2_SCAN_TOTAL_DOSE = "ic2_scan_total_dose"
C_IC3_SCAN_TOTAL_DOSE = "ic3_scan_total_dose"
C_CHARGE_REQ = "charge_req"

# Scan-magnet field probes (gauss) in timeslice_data_device_units.
# G3: r_tx2_probe_x / r_tx2_probe_y (TX2 hall probes).
# G2: field_c_x / field_c_y (correcting-coil field readback).
C_MAG_FIELD_X = "mag_field_x"
C_MAG_FIELD_Y = "mag_field_y"

# Raw position concepts (register-level, need coordinate remap)
C_IC1_X_POS_RAW = "ic1_x_pos_raw"
C_IC1_Y_POS_RAW = "ic1_y_pos_raw"
C_IC2_X_POS_RAW = "ic2_x_pos_raw"
C_IC2_Y_POS_RAW = "ic2_y_pos_raw"

# Non-raw position concepts (already in plan mm coordinates)
C_IC1_X_POS = "ic1_x_pos"
C_IC1_Y_POS = "ic1_y_pos"
C_IC2_X_POS = "ic2_x_pos"
C_IC2_Y_POS = "ic2_y_pos"

# G3 timeslice only: pre-filter strip peak (nA). G2 uses post-filter amplitude.
C_IC1_X_PEAK_AMPLITUDE = "ic1_x_peak_amplitude"
C_IC1_Y_PEAK_AMPLITUDE = "ic1_y_peak_amplitude"
C_IC2_X_PEAK_AMPLITUDE = "ic2_x_peak_amplitude"
C_IC2_Y_PEAK_AMPLITUDE = "ic2_y_peak_amplitude"

IC_PEAK_AMPLITUDE_COLUMNS = (
    C_IC1_X_PEAK_AMPLITUDE,
    C_IC1_Y_PEAK_AMPLITUDE,
    C_IC2_X_PEAK_AMPLITUDE,
    C_IC2_Y_PEAK_AMPLITUDE,
)

# Prescribed plan positions from input_map
C_X_POSITION = "x_position"
C_Y_POSITION = "y_position"

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

_CONCEPT_ALIASES_STATIC: dict[str, tuple[str, ...]] = {
    C_ENERGY: ("ENERGY", "energy", "beam_energy", "energy_mev", "nominal_energy"),
    C_LAYER_ID: ("layer_id", "layer", "layerid", "layer_index", "layer_number"),
    C_SPOT_NO: ("spot_no", "spot", "spot_id", "spot_number", "spot_index"),
    C_TIMESTAMP: ("timestamp", "time_ms", "time_stamp", "timestamp_ms"),
    C_TIME_S: ("time_s", "time_seconds", "time_sec"),
    C_TIME_NS: ("time_ns", "time_nanoseconds", "time_nano"),
    # G3 columns (ic1_primary_channel) are nA; G2 columns (r_ic1_current_dose)
    # are coulombs per 1 ms timeslice — see _COLUMN_SCALE_FACTORS for conversion.
    C_IC1_CURRENT: ("ic1_primary_channel", "ic1", "ic1_current", "ic1_primary", "r_ic1_current_dose", "r_ic1_current_dose_filt"),
    C_IC2_CURRENT: ("ic2_primary_channel", "ic2", "ic2_current", "ic2_primary", "r_ic2_current_dose", "r_ic2_current_dose_filt"),
    C_IC1_STRIP_SUM: ("ic1_strip_sum_channel", "ic1_strip_sum", "ic1_stripsum"),
    C_IC2_STRIP_SUM: ("ic2_strip_sum_channel", "ic2_strip_sum", "ic2_stripsum"),
    C_IC3_CURRENT_A: ("ic3_current_A", "ic3_current_a", "ic3_a_current", "ic3_current_1", "r_px3_1_strip_sum"),
    C_IC3_CURRENT_B: ("ic3_current_B", "ic3_current_b", "ic3_b_current", "ic3_current_2"),
    C_IC3_CURRENT_C: ("ic3_current_C", "ic3_current_c", "ic3_c_current", "ic3_current_3"),
    C_IC3_CURRENT_D: ("ic3_current_D", "ic3_current_d", "ic3_d_current", "ic3_current_4"),
    C_BEAM_CURRENT: ("c_beam_current", "c_beamI", "c_beami", "r_beamI", "r_beami", "beam_current", "rci_beam_current", "beam_i"),
    C_IC1_TOTAL_DOSE: ("ic1_total_dose_spot", "ic1_total_dose_spot_raw", "ic1_total_dose", "ic1_dose_spot_raw"),
    C_IC2_TOTAL_DOSE: ("ic2_total_dose_spot", "ic2_total_dose_spot_raw", "ic2_total_dose", "ic2_dose_spot_raw"),
    C_IC3_TOTAL_DOSE: (
        "r_ic3_total_dose_spot",
        "r_ic3_total_dose_spot_raw",
        "ic3_total_dose_spot_raw",
        "ic3_total_dose",
        "r_ic3_total_dose",
    ),
    C_IC1_SCAN_TOTAL_DOSE: (
        "r_ic1_scan_total_dose",
        "ic1_scan_total_dose",
        "ic1_dose_total",
    ),
    C_IC2_SCAN_TOTAL_DOSE: (
        "r_ic2_scan_total_dose",
        "ic2_scan_total_dose",
        "ic2_dose_total",
    ),
    C_IC3_SCAN_TOTAL_DOSE: (
        "ic3_dose_total",
        "ic3_scan_total_dose",
        "r_ic3_scan_total_dose",
        "r_ic3_dose_total",
    ),
    C_CHARGE_REQ: ("CHARGE_REQ", "charge_req", "dose_req", "prescribed_dose"),
    C_MAG_FIELD_X: (
        "r_tx2_probe_x",
        "field_c_x",
        "mag_field_x",
        "field_x",
        "b_field_x",
    ),
    C_MAG_FIELD_Y: (
        "r_tx2_probe_y",
        "field_c_y",
        "mag_field_y",
        "field_y",
        "b_field_y",
    ),
    C_X_POSITION: ("X_POSITION", "x_position", "xposition", "x_pos", "planned_x"),
    C_Y_POSITION: ("Y_POSITION", "y_position", "yposition", "y_pos", "planned_y"),
    C_IC1_X_PEAK_AMPLITUDE: ("r_ic1_x_peak_amplitude", "ic1_x_peak_amplitude"),
    C_IC1_Y_PEAK_AMPLITUDE: ("r_ic1_y_peak_amplitude", "ic1_y_peak_amplitude"),
    C_IC2_X_PEAK_AMPLITUDE: ("r_ic2_x_peak_amplitude", "ic2_x_peak_amplitude"),
    C_IC2_Y_PEAK_AMPLITUDE: ("r_ic2_y_peak_amplitude", "ic2_y_peak_amplitude"),
}


# G2 IC current columns store charge (coulombs) accumulated over a 1 ms
# timeslice.  To convert to nA:  nA = C / 0.001 s × 1e9 = C × 1e12.
# Keys are normalized column names (lowercase, non-alnum collapsed to "_").
_COLUMN_SCALE_FACTORS: dict[str, float] = {
    "r_ic1_current_dose":      1e12,  # G2 IC1 current — coulombs → nA
    "r_ic1_current_dose_filt": 1e12,  # G2 IC1 filtered current — coulombs → nA
    "r_ic2_current_dose":      1e12,  # G2 IC2 current — coulombs → nA
    "r_ic2_current_dose_filt": 1e12,  # G2 IC2 filtered current — coulombs → nA
    "r_px3_1_strip_sum":       1e12,  # G2 PX3 (IC3 equivalent) — coulombs → nA
}

# G2 per-axis strip sums that need to be summed into a single canonical column.
# Each entry: (target_canonical, [raw_source_columns], scale_factor).
# Only applied when target doesn't already exist (G3 provides it natively).
_DERIVED_SUM_COLUMNS: list[tuple[str, list[str], float]] = [
    (C_IC1_STRIP_SUM, ["r_ic1_x_strip_sum", "r_ic1_y_strip_sum"], 1e12),
    (C_IC2_STRIP_SUM, ["r_ic2_x_strip_sum", "r_ic2_y_strip_sum"], 1e12),
]

# When G2 PX3 maps to ic3_current_a, the B/C/D quad channels don't exist.
# Fill them with 0 so downstream code that sums A+B+C+D works transparently.
_IC3_QUAD_ZERO_FILL = [C_IC3_CURRENT_B, C_IC3_CURRENT_C, C_IC3_CURRENT_D]


def _position_key_variants(position_key: str) -> tuple[str, ...]:
    base = position_key[:-4] if position_key.endswith("_raw") else position_key
    if base == position_key:
        return (position_key,)
    return (position_key, base)


_RAW_POS_CONCEPTS = {C_IC1_X_POS_RAW, C_IC1_Y_POS_RAW, C_IC2_X_POS_RAW, C_IC2_Y_POS_RAW}
_NONRAW_POS_CONCEPTS = {C_IC1_X_POS, C_IC1_Y_POS, C_IC2_X_POS, C_IC2_Y_POS}

_POS_CONCEPT_IC_AXIS: dict[str, tuple[str, str]] = {
    C_IC1_X_POS_RAW: ("ic1", "x"),
    C_IC1_Y_POS_RAW: ("ic1", "y"),
    C_IC2_X_POS_RAW: ("ic2", "x"),
    C_IC2_Y_POS_RAW: ("ic2", "y"),
    C_IC1_X_POS: ("ic1", "x"),
    C_IC1_Y_POS: ("ic1", "y"),
    C_IC2_X_POS: ("ic2", "x"),
    C_IC2_Y_POS: ("ic2", "y"),
}


def concept_column_candidates(concept: str, *, position_key: str | None = None) -> tuple[str, ...]:
    """Return candidate physical column names for a logical concept."""
    if concept in _CONCEPT_ALIASES_STATIC:
        return _CONCEPT_ALIASES_STATIC[concept]

    if position_key is None or concept not in _POS_CONCEPT_IC_AXIS:
        return ()

    ic, axis = _POS_CONCEPT_IC_AXIS[concept]
    is_raw = concept in _RAW_POS_CONCEPTS

    if is_raw:
        # Raw concepts must not pick up non-raw key variants (e.g. spot_position)
        # or canonicalization renames processed iso columns into *_raw names.
        keys = (position_key,)
    else:
        base = position_key[:-4] if position_key.endswith("_raw") else position_key
        keys = (base,)

    return tuple(
        name
        for key in keys
        for name in (f"r_{ic}_{axis}_{key}", f"{ic}_{axis}_{key}")
    )


def normalize_column_name(name: str) -> str:
    """Normalize column names for robust case/spacing/punctuation matching."""
    return _NON_ALNUM_RE.sub("_", str(name).strip().lower()).strip("_")


def resolve_column_name(columns: Iterable[str], requested: str) -> str | None:
    """Resolve a requested name against exact and normalized matching."""
    columns_list = list(columns)
    if requested in columns_list:
        return requested
    normalized_map = {normalize_column_name(c): c for c in columns_list}
    return normalized_map.get(normalize_column_name(requested))


def resolve_concept_column(
    columns: Iterable[str],
    concept: str,
    *,
    position_key: str | None = None,
) -> str | None:
    """Resolve the best matching column for a concept.

    Tries the canonical name first, then falls back to alias candidates.
    """
    resolved = resolve_column_name(columns, concept)
    if resolved is not None:
        return resolved
    for candidate in concept_column_candidates(concept, position_key=position_key):
        resolved = resolve_column_name(columns, candidate)
        if resolved is not None:
            return resolved
    return None


def resolve_requested_column(columns: Iterable[str], requested: str) -> str | None:
    """Resolve an explicit requested column name with compatibility variants."""
    resolved = resolve_column_name(columns, requested)
    if resolved is not None:
        return resolved
    if requested.startswith("r_"):
        return resolve_column_name(columns, requested[2:])
    return resolve_column_name(columns, f"r_{requested}")


def canonical_column_aliases() -> dict[str, tuple[str, ...]]:
    """Build canonical-column alias map used for DataFrame normalization."""
    aliases = {
        canonical: alt_names
        for canonical, alt_names in _CONCEPT_ALIASES_STATIC.items()
    }
    all_pos_concepts = _RAW_POS_CONCEPTS | _NONRAW_POS_CONCEPTS
    for key in (POSITION_KEY_G2_RAW, POSITION_KEY_G3_RAW):
        for concept in all_pos_concepts:
            candidates = concept_column_candidates(concept, position_key=key)
            if candidates:
                aliases[candidates[0]] = tuple(candidates[1:])
    return aliases


def canonicalize_dataframe_columns(
    df: pd.DataFrame,
    *,
    aliases: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """Strip/normalize columns and rename known aliases to canonical names."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    alias_map = aliases or canonical_column_aliases()

    existing = list(out.columns)
    normalized_existing = {normalize_column_name(c): c for c in existing}
    rename_map: dict[str, str] = {}

    for canonical, alt_names in alias_map.items():
        if canonical in existing:
            continue
        for candidate in (canonical, *alt_names):
            actual = normalized_existing.get(normalize_column_name(candidate))
            if actual is None or actual in rename_map:
                continue
            if canonical in existing or canonical in rename_map.values():
                continue
            rename_map[actual] = canonical
            break

    if rename_map:
        out = out.rename(columns=rename_map)

        for original, canonical in rename_map.items():
            factor = _COLUMN_SCALE_FACTORS.get(normalize_column_name(original))
            if factor is not None and factor != 1.0:
                out[canonical] = pd.to_numeric(out[canonical], errors="coerce") * factor

    # Synthesize derived columns (e.g. G2 per-axis strip sums → combined)
    cur_cols = set(out.columns)
    norm_cols = {normalize_column_name(c): c for c in cur_cols}

    for target, sources, factor in _DERIVED_SUM_COLUMNS:
        if target in cur_cols:
            continue
        resolved = [norm_cols.get(normalize_column_name(s)) for s in sources]
        if all(r is not None for r in resolved):
            combined = sum(
                pd.to_numeric(out[r], errors="coerce") for r in resolved
            )
            if factor != 1.0:
                combined = combined * factor
            out[target] = combined

    # Zero-fill IC3 B/C/D when only A exists (G2 PX3 → ic3_current_a)
    if C_IC3_CURRENT_A in out.columns:
        for quad in _IC3_QUAD_ZERO_FILL:
            if quad not in out.columns:
                out[quad] = 0.0

    return out
