"""Session-list Ext./Lyr. fallback from input_map / spot_data."""

from __future__ import annotations

import zipfile
from pathlib import Path

from scan_kit.common.session_source import (
    clear_termination_summary_cache,
    load_session_list_meta,
)

_SUMMARY = """\
Date: Thu Sep 10 21:07:41 2026
Primary total dose: 12.5 MU
Treatment time: 10 seconds
Room number: 2
Configuration name: facility_map2map
"""

_SUMMARY_WITH_GEOM = """\
Date: Thu Sep 10 21:07:41 2026
Layer delivery: 76/76
Spot extent width: 0 mm
Spot extent height: 0 mm
Configuration name: facility_map2map
"""


def _write_session(
    folder: Path,
    sid: str,
    *,
    summary: str = _SUMMARY,
    input_map: str,
    spot_data: str | None = None,
) -> Path:
    root = folder / sid
    root.mkdir()
    (root / "termination_summary.txt").write_text(summary, encoding="utf-8")
    (root / "input_map.csv").write_text(input_map, encoding="utf-8")
    if spot_data is not None:
        (root / "spot_data.csv").write_text(spot_data, encoding="utf-8")
    return root


def test_input_map_fills_missing_summary_geom(tmp_path: Path) -> None:
    clear_termination_summary_cache()
    sid = "geom1"
    root = _write_session(
        tmp_path,
        sid,
        input_map=(
            "ENERGY,X_POSITION,Y_POSITION,layer_id\n"
            "70,-50,-40,999\n"
            "70,50,40,999\n"
            "100,0,0,999\n"
        ),
        spot_data="ENERGY,X_POSITION,Y_POSITION\n1,0,0\n2,999,999\n",
    )
    meta = load_session_list_meta(sid, root)
    assert meta is not None
    assert meta.layer_count == 2
    assert meta.map_extent_mm == 100.0
    assert meta.config_name == "facility_map2map"


def test_zero_summary_extent_not_overwritten_by_map(tmp_path: Path) -> None:
    clear_termination_summary_cache()
    sid = "geom0"
    root = _write_session(
        tmp_path,
        sid,
        summary=_SUMMARY_WITH_GEOM,
        input_map="ENERGY,X_POSITION,Y_POSITION\n70,-50,-50\n70,50,50\n",
    )
    meta = load_session_list_meta(sid, root)
    assert meta is not None
    assert meta.map_extent_mm == 0.0
    assert meta.layer_count == 76


def test_spot_data_used_when_input_map_lacks_xy(tmp_path: Path) -> None:
    clear_termination_summary_cache()
    sid = "geom2"
    root = _write_session(
        tmp_path,
        sid,
        input_map="a,b\n1,2\n",
        spot_data="ENERGY,X_POSITION,Y_POSITION\n70,0,0\n80,10,5\n80,-10,5\n",
    )
    meta = load_session_list_meta(sid, root)
    assert meta is not None
    assert meta.layer_count == 2
    assert meta.map_extent_mm == 20.0


def test_list_meta_zip_does_not_extract(tmp_path: Path) -> None:
    clear_termination_summary_cache()
    sid = "424244"
    archive = tmp_path / f"{sid}.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{sid}/termination_summary.txt", _SUMMARY)
        zf.writestr(
            f"{sid}/input_map.csv",
            "ENERGY,X_POSITION,Y_POSITION\n70,-25,-10\n120,25,10\n",
        )
    meta = load_session_list_meta(sid, archive)
    assert meta is not None
    assert meta.layer_count == 2
    assert meta.map_extent_mm == 50.0
    assert not (tmp_path / sid).exists()
