"""Rebuild G3 ``spot_data.csv`` from per-device layer CSVs (map2map / DCS rules)."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...common.g3_timeslice_position import (
    IsoAxisTransform,
    build_g3_iso_plan_lookup,
    derive_g3_iso_transform,
    plan_has_position_span,
)

# Device strip a/b → G3 x/y raw names (Output.xml: B=X, A=Y; fixture 1091134775).
_IX256_1_RENAME = {
    "spot_no": "spot_no",
    "layer_id": "layer_id",
    "timeslice_number": "timesliceNumber",
    "timeslice_timestamp(ms)": "timestamp",
    "point_time(ms)": "point_time(ms)",
    "ic1_position_measured_b": "r_ic1_x_spot_position_raw",
    "ic1_position_measured_a": "r_ic1_y_spot_position_raw",
    "ic1_sigma_measured_b(mm)": "r_ic1_x_spot_sigma_raw",
    "ic1_sigma_measured_a(mm)": "r_ic1_y_spot_sigma_raw",
    "total_dose(nC)": "ic1_total_dose_spot_raw",
    "ic1_position_b_min": "_ic1_x_min",
    "ic1_position_b_max": "_ic1_x_max",
    "ic1_position_a_min": "_ic1_y_min",
    "ic1_position_a_max": "_ic1_y_max",
}
_IX256_2_RENAME = {
    "spot_no": "spot_no",
    "layer_id": "layer_id",
    "timeslice_number": "timesliceNumber",
    "timeslice_timestamp(ms)": "timestamp",
    "point_time(ms)": "point_time(ms)",
    "ic2_position_measured_b": "r_ic2_x_spot_position_raw",
    "ic2_position_measured_a": "r_ic2_y_spot_position_raw",
    "ic2_sigma_measured_b(mm)": "r_ic2_x_spot_sigma_raw",
    "ic2_sigma_measured_a(mm)": "r_ic2_y_spot_sigma_raw",
    "total_dose(nC)": "ic2_total_dose_spot_raw",
    "ic2_position_b_min": "_ic2_x_min",
    "ic2_position_b_max": "_ic2_x_max",
    "ic2_position_a_min": "_ic2_y_min",
    "ic2_position_a_max": "_ic2_y_max",
}
_FX4_RENAME = {
    "spot_no": "spot_no",
    "layer_id": "layer_id",
    "timeslice_number": "timesliceNumber",
    "timeslice_timestamp(ms)": "timestamp",
    "point_time(ms)": "point_time(ms)",
    "total_dose(nC)": "r_ic3_total_dose_spot_raw",
}
_RCI_RENAME = {
    "spot_no": "spot_no",
    "layer_id": "layer_id",
    "timeslice_number": "timesliceNumber",
    "timeslice_timestamp(ms)": "timestamp",
    "point_time(ms)": "point_time(ms)",
    "beam_current_command": "c_current_rci",
}

_RUN_SPOT_SOURCES = (
    ("IX256_1_spot_data.csv", _IX256_1_RENAME),
    ("IX256_2_spot_data.csv", _IX256_2_RENAME),
    ("FX4_spot_data.csv", _FX4_RENAME),
    ("RCI_spot_data.csv", _RCI_RENAME),
)

_POSITION_RAW_TO_ISO = (
    ("r_ic1_x_spot_position_raw", "r_ic1_x_spot_position", "IC_1_X"),
    ("r_ic1_y_spot_position_raw", "r_ic1_y_spot_position", "IC_1_Y"),
    ("r_ic2_x_spot_position_raw", "r_ic2_x_spot_position", "IC_2_X"),
    ("r_ic2_y_spot_position_raw", "r_ic2_y_spot_position", "IC_2_Y"),
)
_SIGMA_RAW_TO_ISO = (
    ("r_ic1_x_spot_sigma_raw", "r_ic1_x_spot_sigma", "IC_1_X"),
    ("r_ic1_y_spot_sigma_raw", "r_ic1_y_spot_sigma", "IC_1_Y"),
    ("r_ic2_x_spot_sigma_raw", "r_ic2_x_spot_sigma", "IC_2_X"),
    ("r_ic2_y_spot_sigma_raw", "r_ic2_y_spot_sigma", "IC_2_Y"),
)

_AFFINE_AXIS_MAP = (
    ("ic1_x", "r_ic1_x_spot_position_raw", "r_ic1_x_spot_position", "r_ic1_x_spot_sigma_raw", "r_ic1_x_spot_sigma"),
    ("ic1_y", "r_ic1_y_spot_position_raw", "r_ic1_y_spot_position", "r_ic1_y_spot_sigma_raw", "r_ic1_y_spot_sigma"),
    ("ic2_x", "r_ic2_x_spot_position_raw", "r_ic2_x_spot_position", "r_ic2_x_spot_sigma_raw", "r_ic2_x_spot_sigma"),
    ("ic2_y", "r_ic2_y_spot_position_raw", "r_ic2_y_spot_position", "r_ic2_y_spot_sigma_raw", "r_ic2_y_spot_sigma"),
)

_HELPER_COLS = (
    "_ic1_x_min",
    "_ic1_x_max",
    "_ic1_y_min",
    "_ic1_y_max",
    "_ic2_x_min",
    "_ic2_x_max",
    "_ic2_y_min",
    "_ic2_y_max",
)


@dataclass(frozen=True)
class ChamberGeometry:
    """Ion-chamber strip→iso geometry from ``devices.xml``."""

    name: str
    strip_count: float
    strip_to_mm: float
    zero_offset_at_iso_mm: float
    reverse_strips: bool
    sdd_mm: float
    sad_mm: float

    @property
    def center_strip(self) -> float:
        # ponytail: zero_offset_mm is NOT folded into center for G3 fixture match.
        return (self.strip_count - 1.0) / 2.0

    @property
    def scale(self) -> float:
        """Strip→iso mm slope (includes reverse sign)."""
        sign = -1.0 if self.reverse_strips else 1.0
        return sign * self.strip_to_mm * (self.sad_mm / self.sdd_mm)

    def strip_to_iso(self, strip: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(strip, dtype=float) - self.center_strip) + (
            self.zero_offset_at_iso_mm
        )

    def sigma_to_iso(self, sigma_raw: np.ndarray) -> np.ndarray:
        return abs(self.scale) * np.asarray(sigma_raw, dtype=float)


def _read_spot_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(path, index_col=False, skipinitialspace=True)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    return None if frame.empty else frame


def _project_spot_file(path: Path, rename: dict[str, str]) -> pd.DataFrame | None:
    frame = _read_spot_csv(path)
    if frame is None:
        return None
    have = {src: dst for src, dst in rename.items() if src in frame.columns}
    if "spot_no" not in have or "layer_id" not in have:
        return None
    return frame[list(have)].rename(columns=have)


def merge_run_spot_files(run_dir: Path) -> pd.DataFrame | None:
    """Wide-join device spot CSVs in one ``layer-*/run-*`` on ``(layer_id, spot_no)``."""
    merged: pd.DataFrame | None = None
    for name, rename in _RUN_SPOT_SOURCES:
        part = _project_spot_file(run_dir / name, rename)
        if part is None:
            continue
        if merged is None:
            merged = part
            continue
        keys = [c for c in ("spot_no", "layer_id") if c in part.columns and c in merged.columns]
        extra = [c for c in part.columns if c not in merged.columns]
        if not keys or not extra:
            continue
        merged = merged.merge(part[keys + extra], on=keys, how="outer")
    return merged


def reconstruct_spot_dataframe(session_dir: Path) -> pd.DataFrame | None:
    """Join all layer/run device CSVs and attach processed iso columns when possible."""
    frames: list[pd.DataFrame] = []
    for layer_dir in sorted(Path(session_dir).glob("layer-*")):
        if not layer_dir.is_dir():
            continue
        for run_dir in sorted(layer_dir.glob("run-*")):
            frame = merge_run_spot_files(run_dir)
            if frame is not None and not frame.empty:
                frames.append(frame)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    apply_iso_columns(out, Path(session_dir))
    drop = [c for c in _HELPER_COLS if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    return out


def build_g3_spot_data(session_dir: Path) -> Path | None:
    """Write root ``spot_data.csv`` in G3 column names if the device omitted it."""
    session_dir = Path(session_dir)
    dest = session_dir / "spot_data.csv"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    out = reconstruct_spot_dataframe(session_dir)
    if out is None:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    return dest


def write_session_info(session_dir: Path, session_id: str) -> Path:
    """Write ``session_info.json`` when the device did not."""
    dest = Path(session_dir) / "session_info.json"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    payload = {
        "DCSState": "FnStateMapComplete",
        "LayerID": "0",
        "Run": "0",
        "SessionID": session_id,
    }
    dest.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    return dest


def apply_iso_columns(frame: pd.DataFrame, session_dir: Path) -> bool:
    """Add processed position/sigma columns. Prefer ``devices.xml``; else plan affine.

    Returns True when any processed position column was written. Does **not**
    copy raw→processed when iso cannot be derived.
    """
    if _apply_iso_from_devices_xml(frame, Path(session_dir) / "config" / "map2map" / "devices.xml"):
        return True
    return _apply_iso_from_plan_affine(frame, Path(session_dir))


def parse_chamber_geometry(devices_xml: Path | str) -> dict[str, ChamberGeometry]:
    """Parse ion-chamber strip→iso parameters from ``devices.xml``."""
    root = ET.parse(devices_xml).getroot()
    raw: dict[str, dict[str, float | bool | str]] = {}
    for chamber in root.iter("ion_chamber"):
        device_el = chamber.find("device")
        if device_el is None:
            continue
        name = device_el.get("name")
        if not name or name not in {"IC_1_X", "IC_1_Y", "IC_2_X", "IC_2_Y"}:
            continue

        def _text(tag: str, default: str) -> str:
            el = chamber.find(tag)
            return (el.text or default).strip() if el is not None else default

        rev_txt = _text("reverse_strips", "0").lower()
        raw[name] = {
            "strip_count": float(_text("strip_count", "128")),
            "strip_to_mm": float(_text("strip_to_mm", "2")),
            "zero_offset_at_iso_mm": float(_text("zero_offset_at_iso_mm", "0")),
            "reverse_strips": rev_txt in {"1", "true", "yes"},
            "sdd_mm": float(_text("source_to_device_distance_mm", "1")),
            "sad_mm": float(_text("source_to_axis_distance_mm", "1")),
        }

    out: dict[str, ChamberGeometry] = {}
    for name, vals in raw.items():
        # ponytail: G3 fixture uses X-chamber SAD for Y magnification (XML Y SAD=2000
        # does not match processed columns; upgrade if scan_dose C++ is available).
        sad = float(vals["sad_mm"])
        if name.endswith("_Y"):
            sibling = name[:-1] + "X"
            if sibling in raw:
                sad = float(raw[sibling]["sad_mm"])
        out[name] = ChamberGeometry(
            name=name,
            strip_count=float(vals["strip_count"]),
            strip_to_mm=float(vals["strip_to_mm"]),
            zero_offset_at_iso_mm=float(vals["zero_offset_at_iso_mm"]),
            reverse_strips=bool(vals["reverse_strips"]),
            sdd_mm=float(vals["sdd_mm"]),
            sad_mm=sad,
        )
    return out


def _apply_iso_from_devices_xml(frame: pd.DataFrame, devices_xml: Path) -> bool:
    if not devices_xml.is_file():
        return False
    try:
        chambers = parse_chamber_geometry(devices_xml)
    except (ET.ParseError, OSError, TypeError, ValueError):
        return False
    if not chambers:
        return False

    wrote = False
    for raw_col, iso_col, chamber_name in _POSITION_RAW_TO_ISO:
        geom = chambers.get(chamber_name)
        if geom is None or raw_col not in frame.columns:
            continue
        frame[iso_col] = geom.strip_to_iso(frame[raw_col].to_numpy(dtype=float))
        wrote = True

    for raw_col, iso_col, chamber_name in _SIGMA_RAW_TO_ISO:
        geom = chambers.get(chamber_name)
        if geom is None or raw_col not in frame.columns:
            continue
        frame[iso_col] = geom.sigma_to_iso(frame[raw_col].to_numpy(dtype=float))

    return wrote


def _midpoint_series(frame: pd.DataFrame, lo: str, hi: str) -> pd.Series | None:
    if lo not in frame.columns or hi not in frame.columns:
        return None
    return (frame[lo].astype(float) + frame[hi].astype(float)) / 2.0


def _device_targets_from_frame(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Build strip targets for plan-affine iso from min/max midpoints."""
    need = {
        "ic1_x_target": _midpoint_series(frame, "_ic1_x_min", "_ic1_x_max"),
        "ic1_y_target": _midpoint_series(frame, "_ic1_y_min", "_ic1_y_max"),
        "ic2_x_target": _midpoint_series(frame, "_ic2_x_min", "_ic2_x_max"),
        "ic2_y_target": _midpoint_series(frame, "_ic2_y_min", "_ic2_y_max"),
    }
    if any(v is None for v in need.values()):
        return None
    if "spot_no" not in frame.columns or "layer_id" not in frame.columns:
        return None
    out = pd.DataFrame(
        {
            "layer_id": frame["layer_id"].to_numpy(dtype=float),
            "spot_no": frame["spot_no"].to_numpy(dtype=float),
            **{k: v.to_numpy(dtype=float) for k, v in need.items()},  # type: ignore[union-attr]
        }
    )
    return out.drop_duplicates(subset=["layer_id", "spot_no"], keep="first")


def _apply_iso_from_plan_affine(frame: pd.DataFrame, session_dir: Path) -> bool:
    input_map_path = session_dir / "input_map.csv"
    if not input_map_path.is_file():
        return False
    try:
        input_map = pd.read_csv(input_map_path, index_col=False, skipinitialspace=True)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    plan = build_g3_iso_plan_lookup(input_map)
    if plan is None or not plan_has_position_span(plan):
        return False

    device_targets = _device_targets_from_frame(frame)
    if device_targets is None:
        return False

    transform = derive_g3_iso_transform(device_targets, plan)
    if transform is None:
        return False

    axes: dict[str, IsoAxisTransform] = {
        "ic1_x": transform.ic1_x,
        "ic1_y": transform.ic1_y,
        "ic2_x": transform.ic2_x,
        "ic2_y": transform.ic2_y,
    }
    wrote = False
    for axis, raw_pos, iso_pos, raw_sig, iso_sig in _AFFINE_AXIS_MAP:
        aff = axes[axis]
        if not math.isfinite(aff.slope) or abs(aff.slope) < 1e-9:
            continue
        if raw_pos in frame.columns:
            strip = frame[raw_pos].to_numpy(dtype=float)
            frame[iso_pos] = aff.slope * strip + aff.intercept
            wrote = True
        if raw_sig in frame.columns:
            frame[iso_sig] = abs(aff.slope) * frame[raw_sig].to_numpy(dtype=float)
    return wrote
