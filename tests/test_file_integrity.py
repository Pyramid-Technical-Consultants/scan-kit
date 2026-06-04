"""Tests for Pyramid-compatible .md5 file integrity sidecars."""

from __future__ import annotations

from pathlib import Path

import pytest

from scan_kit.common.file_integrity import (
    HEX_DIGEST_LEN,
    IntegrityStatus,
    commit_file_integrity,
    compute_hex_digest,
    parse_sidecar,
    pack_sidecar,
    pyramid_utc_mtime,
    sidecar_path,
    source_path_from_sidecar,
    verify_file_integrity,
)
from scan_kit.workflows.config_tuning.integrity_view import (
    build_integrity_report,
    build_sidecar_only_report,
    integrity_badge_markup,
    integrity_passed,
    is_integrity_data_file,
)

_FIXTURE_MAP2MAP = (
    Path(__file__).resolve().parent.parent
    / "test_data"
    / "1943968267"
    / "1943968267"
    / "config"
    / "map2map"
)

_KNOWN = {
    "tolerances.xml": ("590c408d80858321baae5d070f384f85", 0x04255B44),
    "Input.xml": ("7c132b67956e13f28ab75081d22982a2", 0xDD0C1658),
    "Database.xml": ("e0f6de7f3cc46c2a194f5bc7b8a0b63d", 0xDA293505),
    "devices.xml": ("0d15ddb0df181a9696b1c3a8233c8eff", 0xC70EF424),
}


@pytest.mark.parametrize("xml_name", list(_KNOWN))
def test_verify_session_fixtures(xml_name: str) -> None:
    xml_path = _FIXTURE_MAP2MAP / xml_name
    if not xml_path.is_file():
        pytest.skip(f"missing fixture {xml_path}")

    result = verify_file_integrity(xml_path)
    assert result.status == IntegrityStatus.OK, xml_name
    assert result.sidecar is not None
    assert len(result.sidecar.hex_digest) == HEX_DIGEST_LEN


@pytest.mark.parametrize("xml_name,expected_digest", [(k, v[0]) for k, v in _KNOWN.items()])
def test_known_digest_recomputes(xml_name: str, expected_digest: str) -> None:
    xml_path = _FIXTURE_MAP2MAP / xml_name
    if not xml_path.is_file():
        pytest.skip(f"missing fixture {xml_path}")

    sidecar = parse_sidecar(sidecar_path(xml_path))
    assert sidecar.hex_digest == expected_digest
    assert compute_hex_digest(xml_path.read_bytes(), sidecar.salt) == expected_digest


def test_sidecar_path_suffix() -> None:
    assert sidecar_path("config/map2map/devices.xml").name == "devices.xml.md5"


def test_parse_pack_roundtrip(tmp_path: Path) -> None:
    md5 = tmp_path / "sample.xml.md5"
    md5.write_bytes(pack_sidecar("a" * 32, 12345, 0x6A1FD124))
    assert len(md5.read_bytes()) == 52
    parsed = parse_sidecar(md5)
    assert parsed.hex_digest == "a" * 32
    assert parsed.salt == 12345
    assert parsed.stored_mtime == 0x6A1FD124


def test_commit_then_verify_roundtrip(tmp_path: Path) -> None:
    xml = tmp_path / "sample.xml"
    xml.write_text('<?xml version="1.0"?>\n<root/>\n', encoding="utf-8")

    md5_path = commit_file_integrity(xml)
    assert md5_path == sidecar_path(xml)
    assert md5_path.is_file()

    result = verify_file_integrity(xml)
    assert result.status == IntegrityStatus.OK


def test_commit_deterministic_salt(tmp_path: Path) -> None:
    xml = tmp_path / "sample.xml"
    xml.write_bytes(b"<x/>\n")

    commit_file_integrity(xml, salt=999)
    sidecar = parse_sidecar(sidecar_path(xml))
    assert sidecar.salt == 999
    assert sidecar.hex_digest == compute_hex_digest(b"<x/>\n", 999)


def test_verify_hash_mismatch(tmp_path: Path) -> None:
    xml = tmp_path / "sample.xml"
    xml.write_bytes(b"original\n")
    commit_file_integrity(xml, salt=1)

    xml.write_bytes(b"tampered\n")
    result = verify_file_integrity(xml)
    assert result.status == IntegrityStatus.HASH_ERR
    assert result.expected_digest is not None


def test_verify_missing_sidecar(tmp_path: Path) -> None:
    xml = tmp_path / "sample.xml"
    xml.write_bytes(b"<x/>\n")
    assert verify_file_integrity(xml).status == IntegrityStatus.HASH_FILE_NOT_EXIST


def test_is_integrity_data_file() -> None:
    assert is_integrity_data_file("devices.xml")
    assert not is_integrity_data_file("devices.xml.md5")
    assert not is_integrity_data_file("readme.txt")


def test_integrity_badge_markup() -> None:
    ok_glyph, ok_color, _ = integrity_badge_markup(IntegrityStatus.OK)
    bad_glyph, bad_color, _ = integrity_badge_markup(IntegrityStatus.HASH_ERR)
    assert ok_glyph == "✓"
    assert bad_glyph == "✗"
    assert ok_color != bad_color
    assert integrity_passed(IntegrityStatus.OK)
    assert not integrity_passed(IntegrityStatus.HASH_FILE_NOT_EXIST)


def test_source_path_from_sidecar() -> None:
    assert source_path_from_sidecar("devices.xml.md5").name == "devices.xml"
    assert source_path_from_sidecar("/cfg/map2map/Input.xml.md5").name == "Input.xml"


def test_build_integrity_report_fixture() -> None:
    xml_path = _FIXTURE_MAP2MAP / "tolerances.xml"
    if not xml_path.is_file():
        pytest.skip("missing fixture")
    report = build_integrity_report(xml_path)
    assert report.status == IntegrityStatus.OK
    assert "Salt:" in "\n".join(report.detail_lines)
    assert report.sidecar_path == sidecar_path(xml_path)


def test_build_sidecar_only_report_fixture() -> None:
    md5_path = _FIXTURE_MAP2MAP / "tolerances.xml.md5"
    if not md5_path.is_file():
        pytest.skip("missing fixture")
    report = build_sidecar_only_report(md5_path)
    assert report.status == IntegrityStatus.OK


def test_pyramid_utc_mtime_matches_fixture_tail() -> None:
    xml_path = _FIXTURE_MAP2MAP / "tolerances.xml"
    if not xml_path.is_file():
        pytest.skip("missing fixture")

    sidecar = parse_sidecar(sidecar_path(xml_path))
    assert pyramid_utc_mtime(xml_path) == sidecar.stored_mtime
