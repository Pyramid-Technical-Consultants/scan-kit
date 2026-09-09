"""Tests for G3 session spot_data reconstruction from device layer CSVs."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scan_kit.common.processing import clear_session_raw_cache, load_session_raw
from scan_kit.common.session_source import resolve_session_source
from scan_kit.data.spot import spot_has_ic_positions
from scan_kit.common.devices_xml import IC_SIGMA_DEVICES, load_map2map_geometry
from scan_kit.workflows.plan_runner.g3_session import (
    apply_iso_columns,
    build_g3_spot_data,
    merge_run_spot_files,
    reconstruct_spot_dataframe,
    write_session_info,
)
from scan_kit.workflows.plan_runner.session_packager import package_session_zip

_ROOT = Path(__file__).resolve().parents[1]
_G3 = _ROOT / "test_data" / "1091134775" / "1091134775"
_G3_SPOT = _G3 / "spot_data.csv"
_G3_CONFIG = _G3 / "config" / "map2map"
_G3_DEVICES = _G3_CONFIG / "devices.xml"
_G3_SYSTEM = _G3_CONFIG / "scan_dose_system.xml"
_G3_RUN = _G3 / "layer-0" / "run-0"

# Second room: virtual SAD 2015.36 while every per-IC source_to_axis_distance_mm
# says 1900 / 1520, so only the virtual SAD reproduces the delivered columns.
_ROOM_B = _ROOT / "test_data" / "1307573499" / "1307573499"
_ROOM_B_SPOT = _ROOM_B / "spot_data.csv"
_ROOM_B_SYSTEM = _ROOM_B / "config" / "map2map" / "scan_dose_system.xml"

_MAX_ISO_RESID_MM = 0.05


def _mini_devices_xml() -> str:
    """Minimal IC geometry (matches fixture IC_1/IC_2 X/Y)."""
    return """<?xml version="1.0"?>
<devices>
  <ion_chamber>
    <device name="IC_1_X"/>
    <strip_count>128</strip_count>
    <strip_to_mm>2</strip_to_mm>
    <zero_offset_at_iso_mm>-4.0</zero_offset_at_iso_mm>
    <source_to_device_distance_mm>1496.737</source_to_device_distance_mm>
    <source_to_axis_distance_mm>2500</source_to_axis_distance_mm>
  </ion_chamber>
  <ion_chamber>
    <device name="IC_1_Y"/>
    <strip_count>128</strip_count>
    <strip_to_mm>2</strip_to_mm>
    <zero_offset_at_iso_mm>0.3</zero_offset_at_iso_mm>
    <reverse_strips>1</reverse_strips>
    <source_to_device_distance_mm>1297.757</source_to_device_distance_mm>
    <source_to_axis_distance_mm>2000</source_to_axis_distance_mm>
  </ion_chamber>
  <ion_chamber>
    <device name="IC_2_X"/>
    <strip_count>128</strip_count>
    <strip_to_mm>2</strip_to_mm>
    <zero_offset_at_iso_mm>3.5</zero_offset_at_iso_mm>
    <reverse_strips>1</reverse_strips>
    <source_to_device_distance_mm>1393.2233</source_to_device_distance_mm>
    <source_to_axis_distance_mm>2500</source_to_axis_distance_mm>
  </ion_chamber>
  <ion_chamber>
    <device name="IC_2_Y"/>
    <strip_count>128</strip_count>
    <strip_to_mm>2</strip_to_mm>
    <zero_offset_at_iso_mm>-8.1</zero_offset_at_iso_mm>
    <source_to_device_distance_mm>1179.412</source_to_device_distance_mm>
    <source_to_axis_distance_mm>2000</source_to_axis_distance_mm>
  </ion_chamber>
</devices>
"""


def _mini_system_xml() -> str:
    """``source_to_isocenter_distance`` is the numerator of every IC mag factor."""
    return """<?xml version="1.0"?>
<MapToMap>
  <geometry>
    <source_to_isocenter_distance>2500.0</source_to_isocenter_distance>
    <source_to_x_axis_distance>2500.0</source_to_x_axis_distance>
    <source_to_y_axis_distance>2000.0</source_to_y_axis_distance>
  </geometry>
</MapToMap>
"""


def _write_map2map_config(session_dir: Path) -> Path:
    """Write both files the iso conversion needs: devices + system geometry."""
    cfg = session_dir / "config" / "map2map"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "devices.xml").write_text(_mini_devices_xml(), encoding="utf-8")
    (cfg / "scan_dose_system.xml").write_text(_mini_system_xml(), encoding="utf-8")
    return cfg


def _write_device_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "spot_no": [1, 2],
            "layer_id": [10, 10],
            "ic1_position_measured_a": [64.0, 65.0],
            "ic1_position_measured_b": [64.0, 65.0],
            "ic1_sigma_measured_a(mm)": [1.1, 1.2],
            "ic1_sigma_measured_b(mm)": [2.1, 2.2],
            "total_dose(nC)": [0.3, 0.4],
        }
    ).to_csv(run_dir / "IX256_1_spot_data.csv", index=False)
    pd.DataFrame(
        {
            "spot_no": [1, 2],
            "layer_id": [10, 10],
            "ic2_position_measured_a": [64.0, 65.0],
            "ic2_position_measured_b": [64.0, 65.0],
            "ic2_sigma_measured_a(mm)": [3.1, 3.2],
            "ic2_sigma_measured_b(mm)": [4.1, 4.2],
            "total_dose(nC)": [0.5, 0.6],
        }
    ).to_csv(run_dir / "IX256_2_spot_data.csv", index=False)
    pd.DataFrame(
        {
            "spot_no": [1, 2],
            "layer_id": [10, 10],
            "beam_current_command": [6.1, 6.2],
        }
    ).to_csv(run_dir / "RCI_spot_data.csv", index=False)


@pytest.mark.skipif(
    not (_G3_SPOT.is_file() and _G3_DEVICES.is_file() and _G3_SYSTEM.is_file()),
    reason="G3 fixture missing",
)
def test_reconstruct_fixture_matches_official_iso() -> None:
    built = reconstruct_spot_dataframe(_G3)
    assert built is not None
    official = pd.read_csv(_G3_SPOT, index_col=False, skipinitialspace=True)
    built = built.sort_values(["layer_id", "spot_no"]).reset_index(drop=True)
    official = official.sort_values(["layer_id", "spot_no"]).reset_index(drop=True)
    assert len(built) == len(official)

    for col in (
        "r_ic1_x_spot_position_raw",
        "r_ic1_y_spot_position_raw",
        "r_ic2_x_spot_position_raw",
        "r_ic2_y_spot_position_raw",
    ):
        b = built[col].to_numpy(dtype=float)
        g = official[col].to_numpy(dtype=float)
        both = np.isfinite(b) & np.isfinite(g)
        assert both.any()
        assert float(np.max(np.abs(b[both] - g[both]))) < 1e-4

    for col in (
        "r_ic1_x_spot_position",
        "r_ic1_y_spot_position",
        "r_ic2_x_spot_position",
        "r_ic2_y_spot_position",
    ):
        b = built[col].to_numpy(dtype=float)
        g = official[col].to_numpy(dtype=float)
        raw = built[col + "_raw"].to_numpy(dtype=float)
        both = np.isfinite(b) & np.isfinite(g) & (raw > 1.0) & (g > -100.0) & (g < 500.0)
        assert both.sum() > 100
        resid = float(np.max(np.abs(b[both] - g[both])))
        assert resid < _MAX_ISO_RESID_MM, f"{col} max resid {resid}"


@pytest.mark.skipif(
    not (_ROOM_B_SPOT.is_file() and _ROOM_B_SYSTEM.is_file()),
    reason="second-room fixture missing",
)
def test_iso_columns_match_in_room_with_disagreeing_per_ic_sad() -> None:
    """Only ``source_to_isocenter_distance`` reproduces this room's delivered columns.

    Every chamber here has ``source_to_axis_distance_mm`` well below the room's
    ``source_to_isocenter_distance``, so a per-IC SAD magnification misses by
    millimetres.  This is the regression the 2500 mm fixture cannot catch, because
    there the two happen to agree on the X chambers.
    """
    geometry = load_map2map_geometry(_ROOM_B / "config" / "map2map")
    assert geometry is not None
    for device in IC_SIGMA_DEVICES:
        chamber = geometry.chambers[device]
        assert abs(chamber.xml_sad_mm - geometry.system.virtual_sad_mm) > 100.0

    official = pd.read_csv(_ROOM_B_SPOT, index_col=False, skipinitialspace=True)
    axes = ("ic1_x", "ic1_y", "ic2_x", "ic2_y")
    raw_cols = [f"r_{axis}_spot_position_raw" for axis in axes]
    raw_cols += [f"r_{axis}_spot_sigma_raw" for axis in axes]
    frame = official[raw_cols].copy()
    assert apply_iso_columns(frame, _ROOM_B)

    for axis in axes:
        for col in (f"r_{axis}_spot_position", f"r_{axis}_spot_sigma"):
            got = frame[col].to_numpy(dtype=float)
            expected = official[col].to_numpy(dtype=float)
            both = np.isfinite(got) & np.isfinite(expected)
            assert both.any()
            assert float(np.max(np.abs(got[both] - expected[both]))) < 1e-6, col


@pytest.mark.skipif(not _G3_RUN.is_dir(), reason="G3 fixture missing")
def test_merge_run_raw_axes_match_fixture() -> None:
    merged = merge_run_spot_files(_G3_RUN)
    assert merged is not None
    g3 = pd.read_csv(_G3_SPOT, nrows=3, index_col=False, skipinitialspace=True)
    pd.testing.assert_series_equal(
        merged["r_ic1_x_spot_position_raw"].head(3).reset_index(drop=True),
        g3["r_ic1_x_spot_position_raw"].head(3).reset_index(drop=True),
        check_names=False,
        atol=1e-5,
    )


def test_plan_affine_iso_without_devices_xml(tmp_path: Path) -> None:
    """Spanning plan + strip min/max midpoints → derive_g3_iso_transform path."""
    session_dir = tmp_path / "affine"
    run = session_dir / "layer-0" / "run-0"
    run.mkdir(parents=True)
    n = 20
    spots = list(range(1, n + 1))
    xs = np.linspace(-10.0, 10.0, n)
    ys = np.linspace(-5.0, 5.0, n)
    lines = ["ENERGY,X_POSITION,Y_POSITION,spot_no,layer_id"]
    lines.extend(f"100,{xs[i]},{ys[i]},{spots[i]},1" for i in range(n))
    (session_dir / "input_map.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sx = 63.5 + xs / 3.34
    sy = 63.5 - ys / 3.85
    for prefix, path in (
        ("ic1", run / "IX256_1_spot_data.csv"),
        ("ic2", run / "IX256_2_spot_data.csv"),
    ):
        pd.DataFrame(
            {
                "spot_no": spots,
                "layer_id": [1] * n,
                f"{prefix}_position_measured_a": sy,
                f"{prefix}_position_measured_b": sx,
                f"{prefix}_sigma_measured_a(mm)": [1.0] * n,
                f"{prefix}_sigma_measured_b(mm)": [1.0] * n,
                f"{prefix}_position_a_min": sy - 0.5,
                f"{prefix}_position_a_max": sy + 0.5,
                f"{prefix}_position_b_min": sx - 0.5,
                f"{prefix}_position_b_max": sx + 0.5,
                "total_dose(nC)": [0.1] * n,
            }
        ).to_csv(path, index=False)

    df = reconstruct_spot_dataframe(session_dir)
    assert df is not None
    assert "r_ic1_x_spot_position" in df.columns
    assert np.corrcoef(df["r_ic1_x_spot_position"].to_numpy(float), xs)[0, 1] > 0.99
    assert float(np.max(np.abs(df["r_ic1_x_spot_position"].to_numpy(float) - xs))) < 0.05


def test_omit_processed_without_devices_or_plan_span(tmp_path: Path) -> None:
    session_dir = tmp_path / "flat"
    _write_device_run(session_dir / "layer-0" / "run-0")
    (session_dir / "input_map.csv").write_text(
        "ENERGY,X_POSITION,Y_POSITION,spot_no,layer_id\n100,0,0,1,10\n100,0,0,2,10\n",
        encoding="utf-8",
    )
    path = build_g3_spot_data(session_dir)
    assert path is not None
    df = pd.read_csv(path)
    assert "r_ic1_x_spot_position_raw" in df.columns
    assert "r_ic1_x_spot_position" not in df.columns


def test_devices_xml_writes_iso_not_raw_copy(tmp_path: Path) -> None:
    session_dir = tmp_path / "geom"
    _write_device_run(session_dir / "layer-0" / "run-0")
    _write_map2map_config(session_dir)
    (session_dir / "input_map.csv").write_text(
        "ENERGY,X_POSITION,Y_POSITION,spot_no,layer_id\n100,0,0,1,10\n",
        encoding="utf-8",
    )
    path = build_g3_spot_data(session_dir)
    df = pd.read_csv(path)
    assert "r_ic1_x_spot_position" in df.columns
    # strip 64 → near zero_offset_at_iso (-4) with small strip offset from center 63.5
    assert abs(float(df["r_ic1_x_spot_position"].iloc[0]) - (-4.0 + 3.3406 * 0.5)) < 0.01
    assert float(df["r_ic1_x_spot_position"].iloc[0]) != float(
        df["r_ic1_x_spot_position_raw"].iloc[0]
    )
    assert spot_has_ic_positions(df.columns.tolist())


def test_rebuilt_zip_loads_like_g3_session(tmp_path: Path) -> None:
    session_id = "777001"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    _write_device_run(session_dir / "layer-0" / "run-0")
    _write_map2map_config(session_dir)
    (session_dir / "input_map.csv").write_text(
        "ENERGY,CURRENT,X_POSITION,Y_POSITION,spot_no,layer_id\n"
        "100,1e-9,0,0,1,10\n100,1e-9,1,0,2,10\n",
        encoding="utf-8",
    )
    build_g3_spot_data(session_dir)
    write_session_info(session_dir, session_id)

    zip_path = tmp_path / f"{session_id}.zip"
    package_session_zip(session_dir, zip_path)
    # Also leave directory form discoverable (resolve prefers dir).
    src = resolve_session_source(session_id, tmp_path)
    assert src is not None
    clear_session_raw_cache()
    input_map, spot_data = load_session_raw(session_id, str(tmp_path))
    assert input_map is not None
    assert spot_data is not None
    assert spot_has_ic_positions(spot_data.columns.tolist())
    assert "r_ic1_x_spot_position" in spot_data.columns

    # Zip alone: extract-style resolve via archive helper path
    extract = tmp_path / "from_zip"
    extract.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract)
    clear_session_raw_cache()
    src2 = resolve_session_source(session_id, extract)
    assert src2 is not None
    _, spot2 = load_session_raw(session_id, str(extract))
    assert spot2 is not None
    assert spot_has_ic_positions(spot2.columns.tolist())


@pytest.mark.skipif(not _G3.is_dir(), reason="G3 fixture missing")
def test_fixture_tree_zip_analysis_load(tmp_path: Path) -> None:
    """Package a slim copy of the fixture (layers + map2map + input_map) and load it."""
    session_id = "1091134775"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    shutil.copy2(_G3 / "input_map.csv", session_dir / "input_map.csv")
    cfg = session_dir / "config" / "map2map"
    cfg.mkdir(parents=True)
    shutil.copy2(_G3_DEVICES, cfg / "devices.xml")
    shutil.copy2(_G3_SYSTEM, cfg / "scan_dose_system.xml")
    run_src = _G3 / "layer-0" / "run-0"
    run_dst = session_dir / "layer-0" / "run-0"
    run_dst.mkdir(parents=True)
    for name in (
        "IX256_1_spot_data.csv",
        "IX256_2_spot_data.csv",
        "FX4_spot_data.csv",
        "RCI_spot_data.csv",
    ):
        src = run_src / name
        if src.is_file():
            shutil.copy2(src, run_dst / name)

    assert build_g3_spot_data(session_dir) is not None
    write_session_info(session_dir, session_id)
    package_session_zip(session_dir, tmp_path / f"{session_id}.zip")

    clear_session_raw_cache()
    src = resolve_session_source(session_id, tmp_path)
    assert src is not None
    input_map, spot_data = load_session_raw(session_id, str(tmp_path))
    assert input_map is not None
    assert spot_data is not None
    assert spot_has_ic_positions(spot_data.columns.tolist())
    # processed must not equal raw for IC1 X on fixture geometry
    assert not np.allclose(
        spot_data["r_ic1_x_spot_position"].to_numpy(dtype=float),
        spot_data["r_ic1_x_spot_position_raw"].to_numpy(dtype=float),
    )
