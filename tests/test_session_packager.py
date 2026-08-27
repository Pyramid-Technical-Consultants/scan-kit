"""Tests for plan runner session zip packaging."""

from __future__ import annotations

import zipfile
from pathlib import Path

from scan_kit.common.session_source import session_source_from_archive
from scan_kit.workflows.plan_runner.session_packager import package_session_zip


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
