"""Tests for IGX HTTP URL helpers."""

from scan_kit.igx.http import device_file_url, io_url, normalize_host


def test_normalize_host_adds_default_port() -> None:
    assert normalize_host("192.168.1.1") == "192.168.1.1:80"
    assert normalize_host("192.168.1.1:8080") == "192.168.1.1:8080"


def test_io_url() -> None:
    assert io_url("10.0.0.1", "admin/version", "value.json") == (
        "http://10.0.0.1:80/io/admin/version/value.json"
    )


def test_device_file_url() -> None:
    assert device_file_url("10.0.0.1", "/root/config/control_points.csv") == (
        "http://10.0.0.1:80/root/config/control_points.csv"
    )
