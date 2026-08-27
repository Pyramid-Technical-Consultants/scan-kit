"""Tests for plan runner G3 session zip packaging."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from scan_kit.common.processing import clear_session_raw_cache, load_session_raw
from scan_kit.common.session_source import (
    resolve_session_source,
    session_source_from_archive,
)
from scan_kit.data.spot import spot_has_ic_positions
from scan_kit.workflows.plan_runner.g3_session import (
    build_g3_spot_data,
    write_session_info,
)
from scan_kit.workflows.plan_runner.session_packager import (
    download_session_files,
    download_session_zip,
    package_session_zip,
)

_ROOT = Path(__file__).resolve().parents[1]
_G3_DEVICES = (
    _ROOT / "test_data" / "1091134775" / "1091134775" / "config" / "map2map" / "devices.xml"
)


def _mini_devices_xml() -> str:
    return _G3_DEVICES.read_text(encoding="utf-8") if _G3_DEVICES.is_file() else (
        '<?xml version="1.0"?><devices>'
        '<ion_chamber><device name="IC_1_X"/>'
        "<strip_count>128</strip_count><strip_to_mm>2</strip_to_mm>"
        "<zero_offset_at_iso_mm>-4.0</zero_offset_at_iso_mm>"
        "<source_to_device_distance_mm>1496.737</source_to_device_distance_mm>"
        "<source_to_axis_distance_mm>2500</source_to_axis_distance_mm>"
        "</ion_chamber>"
        '<ion_chamber><device name="IC_1_Y"/>'
        "<strip_count>128</strip_count><strip_to_mm>2</strip_to_mm>"
        "<zero_offset_at_iso_mm>0.3</zero_offset_at_iso_mm>"
        "<reverse_strips>1</reverse_strips>"
        "<source_to_device_distance_mm>1297.757</source_to_device_distance_mm>"
        "<source_to_axis_distance_mm>2000</source_to_axis_distance_mm>"
        "</ion_chamber>"
        '<ion_chamber><device name="IC_2_X"/>'
        "<strip_count>128</strip_count><strip_to_mm>2</strip_to_mm>"
        "<zero_offset_at_iso_mm>3.5</zero_offset_at_iso_mm>"
        "<reverse_strips>1</reverse_strips>"
        "<source_to_device_distance_mm>1393.2233</source_to_device_distance_mm>"
        "<source_to_axis_distance_mm>2500</source_to_axis_distance_mm>"
        "</ion_chamber>"
        '<ion_chamber><device name="IC_2_Y"/>'
        "<strip_count>128</strip_count><strip_to_mm>2</strip_to_mm>"
        "<zero_offset_at_iso_mm>-8.1</zero_offset_at_iso_mm>"
        "<source_to_device_distance_mm>1179.412</source_to_device_distance_mm>"
        "<source_to_axis_distance_mm>2000</source_to_axis_distance_mm>"
        "</ion_chamber>"
        "</devices>"
    )


def _write_device_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "spot_no": [1, 2],
            "layer_id": [10, 10],
            "timeslice_number": [1, 2],
            "timeslice_timestamp(ms)": [1.0, 2.0],
            "point_time(ms)": [4.0, 5.0],
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


def test_package_session_zip_layout(tmp_path: Path) -> None:
    session_id = "demo_session"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    (session_dir / "input_map.csv").write_text("#NO,ENERGY(MeV)\n1,100\n", encoding="utf-8")
    layer_dir = session_dir / "layer-0" / "run-0"
    layer_dir.mkdir(parents=True)
    (layer_dir / "timeslice_data_device_units.csv").write_text("t,v\n0,1\n", encoding="utf-8")

    zip_path = tmp_path / f"{session_id}.zip"
    package_session_zip(session_dir, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert f"{session_id}/input_map.csv" in names
    assert f"{session_id}/layer-0/run-0/timeslice_data_device_units.csv" in names

    src = session_source_from_archive(zip_path)
    assert src is not None
    assert src.session_id == session_id


def test_build_g3_spot_data_from_device_files(tmp_path: Path) -> None:
    session_dir = tmp_path / "12345"
    _write_device_run(session_dir / "layer-0" / "run-0")
    cfg = session_dir / "config" / "map2map"
    cfg.mkdir(parents=True)
    (cfg / "devices.xml").write_text(_mini_devices_xml(), encoding="utf-8")
    (session_dir / "input_map.csv").write_text(
        "ENERGY,X_POSITION,Y_POSITION,spot_no,layer_id\n100,0,0,1,10\n",
        encoding="utf-8",
    )

    path = build_g3_spot_data(session_dir)
    assert path is not None
    assert path.name == "spot_data.csv"
    df = pd.read_csv(path)
    assert list(df["r_ic1_x_spot_position_raw"]) == [64.0, 65.0]
    assert list(df["r_ic1_y_spot_position_raw"]) == [64.0, 65.0]
    assert list(df["r_ic2_x_spot_position_raw"]) == [64.0, 65.0]
    assert list(df["r_ic2_y_spot_position_raw"]) == [64.0, 65.0]
    assert "r_ic1_x_spot_position" in df.columns
    assert float(df["r_ic1_x_spot_position"].iloc[0]) != 64.0
    assert list(df["c_current_rci"]) == [6.1, 6.2]
    assert spot_has_ic_positions(df.columns.tolist())

    again = build_g3_spot_data(session_dir)
    assert again == path


def test_synthesized_zip_opens_in_session_tools(tmp_path: Path) -> None:
    session_id = "555001"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    (session_dir / "input_map.csv").write_text(
        "ENERGY,CURRENT,X_POSITION,Y_POSITION,spot_no,layer_id\n"
        "100,1e-9,0,0,1,10\n100,1e-9,1,0,2,10\n",
        encoding="utf-8",
    )
    run = session_dir / "layer-0" / "run-0"
    _write_device_run(run)
    cfg = session_dir / "config" / "map2map"
    cfg.mkdir(parents=True)
    (cfg / "devices.xml").write_text(_mini_devices_xml(), encoding="utf-8")
    (run / "timeslice_data_device_units.csv").write_text(
        "timestamp,ic1_primary_channel\n0,1\n",
        encoding="utf-8",
    )
    build_g3_spot_data(session_dir)
    write_session_info(session_dir, session_id)

    zip_path = tmp_path / f"{session_id}.zip"
    package_session_zip(session_dir, zip_path)

    src = resolve_session_source(session_id, tmp_path)
    assert src is not None
    assert src.kind == "directory"
    assert (src.path / "input_map.csv").is_file()
    assert (src.path / "spot_data.csv").is_file()

    clear_session_raw_cache()
    input_map, spot_data = load_session_raw(session_id, str(tmp_path))
    assert input_map is not None
    assert spot_data is not None
    assert "r_ic1_x_spot_position" in spot_data.columns


def test_download_session_zip_uses_session_id_prefix(tmp_path: Path, monkeypatch) -> None:
    files = {
        "/root/reports/session/abc/input_map.csv": b"ENERGY,spot_no,layer_id\n100,1,1\n",
        "/root/reports/session/abc/config/map2map/devices.xml": _mini_devices_xml().encode(
            "utf-8"
        ),
        "/root/reports/session/abc/layer-0/run-0/IX256_1_spot_data.csv": (
            b"spot_no,layer_id,ic1_position_measured_a,ic1_position_measured_b\n"
            b"1,1,64,64\n"
        ),
        "/root/reports/session/abc/layer-0/run-0/IX256_2_spot_data.csv": (
            b"spot_no,layer_id,ic2_position_measured_a,ic2_position_measured_b\n"
            b"1,1,64,64\n"
        ),
    }

    def _get_bytes(host: str, remote: str) -> bytes:
        del host
        if remote in files:
            return files[remote]
        raise FileNotFoundError(remote)

    monkeypatch.setattr(
        "scan_kit.workflows.plan_runner.session_packager.get_bytes",
        _get_bytes,
    )

    zip_path = tmp_path / "abc.zip"
    download_session_zip("192.168.100.184", "/root/reports/session/abc", zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert "abc/input_map.csv" in names
    assert "abc/spot_data.csv" in names
    assert "abc/session_info.json" in names
    assert "abc/config/map2map/devices.xml" in names
    assert not any(name.startswith(".abc_staging") for name in names)
    assert not any(tmp_path.glob(".*_staging"))

    src = resolve_session_source("abc", tmp_path)
    assert src is not None
    spots = pd.read_csv(src.path / "spot_data.csv")
    assert spots["r_ic1_x_spot_position_raw"].iloc[0] == 64
    assert "r_ic1_x_spot_position" in spots.columns


def test_download_session_files_stops_after_empty_layers(
    tmp_path: Path, monkeypatch
) -> None:
    hits: list[str] = []

    def _get_bytes(host: str, remote: str) -> bytes:
        del host
        hits.append(remote)
        if remote.endswith("input_map.csv"):
            return b"ENERGY\n100\n"
        if "/layer-0/run-0/RCI_spot_data.csv" in remote:
            return b"spot_no,layer_id\n1,1\n"
        raise FileNotFoundError(remote)

    monkeypatch.setattr(
        "scan_kit.workflows.plan_runner.session_packager.get_bytes",
        _get_bytes,
    )
    dest = tmp_path / "sess"
    downloaded = download_session_files("host", "/root/reports/session/s", dest)
    assert any(p.name == "input_map.csv" for p in downloaded)
    assert any("layer-0" in str(p) for p in downloaded)
    layer_hits = [h for h in hits if "/layer-" in h]
    assert not any("/layer-5/" in h for h in layer_hits)
    assert any("config/map2map/devices.xml" in h for h in hits)
