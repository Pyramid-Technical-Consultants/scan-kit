"""Suggested default filenames for exported input_map CSV files."""

from __future__ import annotations

import re
from typing import Any

from .base import PlanTemplate
from .energies import STANDARD_ENERGIES_MEV
from .params import normalize_selected_energies
from .spot_weight import (
    SPOT_WEIGHT_METHOD_EVEN_TOTAL,
    SPOT_WEIGHT_METHOD_FIXED,
    SPOT_WEIGHT_METHOD_LAYER_EVEN,
    SPOT_WEIGHT_METHOD_RANDOM,
    SPOT_WEIGHT_METHOD_RANDOM_TOTAL,
)

DEFAULT_FILENAME_MAX_LENGTH = 128
_CSV_SUFFIX = ".csv"

_TEMPLATE_SLUGS: dict[str, str] = {
    "dicom_rt_plan": "DicomPlan",
    "iba_pld_plan": "IbaPld",
    "zero_field": "ZeroField",
    "rectangular_field": "RectField",
}

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def suggest_input_map_filename(
    template: PlanTemplate,
    params: dict[str, Any],
    *,
    max_length: int = DEFAULT_FILENAME_MAX_LENGTH,
) -> str:
    """Build a descriptive default CSV filename from template + parameters."""
    parts = [
        _template_slug(template),
        _energy_part(template.id, params),
        _geometry_part(template.id, params),
        _weight_part(template.id, params),
    ]
    return _join_and_limit(parts, max_length=max_length)


def _template_slug(template: PlanTemplate) -> str:
    slug = _TEMPLATE_SLUGS.get(template.id)
    if slug:
        return slug
    return _sanitize("_".join(template.name.split()))


def _energy_part(template_id: str, params: dict[str, Any]) -> str:
    if template_id in {"dicom_rt_plan", "iba_pld_plan"}:
        return ""

    energies = normalize_selected_energies(
        params.get("selected_energies"),
        catalog=STANDARD_ENERGIES_MEV,
    )
    if not energies:
        return "E0L"

    if set(energies) == set(STANDARD_ENERGIES_MEV):
        return f"E{STANDARD_ENERGIES_MEV[-1]:g}-{STANDARD_ENERGIES_MEV[0]:g}"

    if len(energies) == 1:
        return f"E{energies[0]:g}"

    catalog_indices = [
        STANDARD_ENERGIES_MEV.index(energy)
        for energy in energies
        if energy in STANDARD_ENERGIES_MEV
    ]
    if len(catalog_indices) == len(energies):
        lo_idx = min(catalog_indices)
        hi_idx = max(catalog_indices)
        if hi_idx - lo_idx + 1 == len(energies):
            hi = STANDARD_ENERGIES_MEV[hi_idx]
            lo = STANDARD_ENERGIES_MEV[lo_idx]
            return f"E{hi:g}-{lo:g}"

    if len(energies) <= 4:
        return "E" + "-".join(f"{energy:g}" for energy in energies)

    return f"E{len(energies)}L_{energies[0]:g}-{energies[-1]:g}"


def _import_plan_label_from_path(path_text: str, *, label_reader: str) -> str:
    if not path_text:
        return ""
    from pathlib import Path

    path = Path(path_text)
    try:
        if label_reader == "dicom":
            from .dicom_rt_plan import rt_plan_label_from_path

            label = rt_plan_label_from_path(path)
        elif label_reader == "pld":
            from .iba_pld_plan import pld_plan_label_from_path

            label = pld_plan_label_from_path(path)
        else:
            label = path.stem
    except Exception:
        label = path.stem
    return _sanitize(label)


def _geometry_part(template_id: str, params: dict[str, Any]) -> str:
    if template_id == "dicom_rt_plan":
        return _import_plan_label_from_path(
            str(params.get("dicom_path", "") or "").strip(),
            label_reader="dicom",
        )

    if template_id == "iba_pld_plan":
        return _import_plan_label_from_path(
            str(params.get("pld_path", "") or "").strip(),
            label_reader="pld",
        )

    if template_id == "zero_field":
        spots = int(params.get("spots_per_layer", 0))
        return f"Sp{spots}"

    if template_id == "rectangular_field":
        segments: list[str] = []
        center_x = float(params.get("center_x_mm", 0.0))
        center_y = float(params.get("center_y_mm", 0.0))
        if center_x != 0.0 or center_y != 0.0:
            segments.append(f"C{_num(center_x)}x{_num(center_y)}")

        width = float(params.get("field_width_mm", 0.0))
        height = float(params.get("field_height_mm", 0.0))
        segments.append(f"{_num(width)}x{_num(height)}mm")

        spots_x = int(params.get("spots_x", 0))
        spots_y = int(params.get("spots_y", 0))
        segments.append(f"G{spots_x}x{spots_y}")
        return "_".join(segments)

    return ""


def _weight_part(template_id: str, params: dict[str, Any]) -> str:
    if template_id in {"dicom_rt_plan", "iba_pld_plan"}:
        return ""

    method = params.get("spot_weight_method", SPOT_WEIGHT_METHOD_FIXED)
    if method == SPOT_WEIGHT_METHOD_FIXED:
        return f"Wfix{_num(float(params.get('spot_weight_mu', 0.0)), decimals=4)}"
    if method == SPOT_WEIGHT_METHOD_RANDOM:
        lo = _num(float(params.get("spot_weight_min_mu", 0.0)), decimals=4)
        hi = _num(float(params.get("spot_weight_max_mu", 0.0)), decimals=4)
        return f"Wrng{lo}-{hi}"
    if method == SPOT_WEIGHT_METHOD_LAYER_EVEN:
        lo = _num(float(params.get("spot_weight_min_mu", 0.0)), decimals=4)
        hi = _num(float(params.get("spot_weight_max_mu", 0.0)), decimals=4)
        prefix = "Wlyrs" if params.get("spot_weight_layer_shuffle") else "Wlyr"
        return f"{prefix}{lo}-{hi}"
    if method == SPOT_WEIGHT_METHOD_EVEN_TOTAL:
        return f"Wtot{_num(float(params.get('spot_weight_total_mu', 0.0)), decimals=4)}"
    if method == SPOT_WEIGHT_METHOD_RANDOM_TOTAL:
        total = _num(float(params.get("spot_weight_total_mu", 0.0)), decimals=4)
        variance = _num(float(params.get("spot_weight_variance_pct", 0.0)), decimals=1)
        return f"Wtot{total}v{variance}"
    return "W"


def _num(value: float, *, decimals: int = 3) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _sanitize(text: str) -> str:
    cleaned = _INVALID_FILENAME_CHARS.sub("", text)
    return cleaned.replace(" ", "_")


def _join_and_limit(parts: list[str], *, max_length: int) -> str:
    stem = _sanitize("_".join(part for part in parts if part))
    if not stem:
        stem = "input_map"

    max_stem = max(1, max_length - len(_CSV_SUFFIX))
    if len(stem) <= max_stem:
        return f"{stem}{_CSV_SUFFIX}"

    compact_parts = parts[:]
    compact_parts[1] = _energy_part_compact(compact_parts[1])
    stem = _sanitize("_".join(part for part in compact_parts if part))
    if len(stem) <= max_stem:
        return f"{stem}{_CSV_SUFFIX}"

    # Drop geometry center/detail first, then weight detail.
    if len(compact_parts) > 2 and compact_parts[2]:
        compact_parts[2] = _geometry_part_compact(compact_parts[2])
        stem = _sanitize("_".join(part for part in compact_parts if part))
        if len(stem) <= max_stem:
            return f"{stem}{_CSV_SUFFIX}"

    minimal = _sanitize("_".join(part for part in (compact_parts[0], compact_parts[1]) if part))
    if len(minimal) <= max_stem:
        return f"{minimal}{_CSV_SUFFIX}"

    return f"{minimal[:max_stem]}{_CSV_SUFFIX}"


def _energy_part_compact(energy_part: str) -> str:
    match = re.fullmatch(r"E(\d+)L_.+", energy_part)
    if match:
        return f"E{match.group(1)}L"
    if energy_part.startswith("E") and "-" in energy_part:
        return energy_part
    if energy_part.startswith("E") and energy_part.count("-") == 0:
        values = energy_part[1:].split("-")
        if len(values) > 1:
            return f"E{len(values)}L"
    return energy_part


def _geometry_part_compact(geometry_part: str) -> str:
    if geometry_part.startswith("Sp"):
        return geometry_part
    segments = geometry_part.split("_")
    kept = [segment for segment in segments if segment.startswith("G")]
    if kept:
        return kept[0]
    return geometry_part.split("_")[-1] if segments else geometry_part
