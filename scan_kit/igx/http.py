"""HTTP REST helpers for IGX /io/* resources and device filesystem paths."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests

DEFAULT_PORT = 80
REQUEST_TIMEOUT = (3.05, 30)


def parse_host(host: str) -> str:
    """Accept IP, host:port, or http(s) URL; return host or host:port."""
    text = host.strip()
    if not text:
        raise ValueError("host is required")
    if "://" in text:
        parsed = urlparse(text)
        if not parsed.hostname:
            raise ValueError(f"invalid host: {host}")
        if parsed.port:
            return f"{parsed.hostname}:{parsed.port}"
        return parsed.hostname
    return text.split("/", 1)[0]


def normalize_host(host: str) -> str:
    """Return host with port; use DEFAULT_PORT if no port in host."""
    host = parse_host(host)
    return f"{host}:{DEFAULT_PORT}" if ":" not in host else host


def path_segment(path: str) -> str:
    """Strip leading and trailing slashes for URL path segment."""
    return path.strip("/").rstrip("/")


def io_url(host: str, path: str, file_name: str) -> str:
    """Build URL for GET/PUT /io/<path>/<file_name>."""
    base = f"http://{normalize_host(host)}/io"
    p = path_segment(path)
    return f"{base}/{p}/{file_name}" if p else f"{base}/{file_name}"


def device_file_url(host: str, device_path: str) -> str:
    """HTTP URL for a device filesystem path (e.g. /root/config/control_points.csv)."""
    p = device_path if device_path.startswith("/") else f"/{device_path}"
    return f"http://{normalize_host(host)}{p}"


def normalize_set_value(value: Any) -> Any:
    """Coerce strings to bool/int/float for JSON PUT bodies."""
    if isinstance(value, list):
        return [normalize_set_value(v) for v in value]
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return float(s) if "." in s or "e" in s.lower() else int(s)
    except ValueError:
        return value


def get_json(host: str, path: str, resource: str = "index.json") -> dict:
    """GET /io/<path>/<resource> and return JSON."""
    resp = requests.get(io_url(host, path, resource), timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def put_json(host: str, path: str, field: str, value: Any) -> None:
    """PUT /io/<path>/<field>.json with JSON body."""
    resp = requests.put(
        io_url(host, path, f"{field}.json"),
        json=normalize_set_value(value),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()


def get_bytes(host: str, device_path: str) -> bytes:
    """GET bytes from a device filesystem path."""
    with requests.get(
        device_file_url(host, device_path),
        timeout=REQUEST_TIMEOUT,
        stream=True,
    ) as resp:
        resp.raise_for_status()
        return resp.content


def put_bytes(host: str, device_path: str, data: bytes) -> None:
    """PUT bytes to a device filesystem path."""
    with requests.put(
        device_file_url(host, device_path),
        data=data,
        timeout=REQUEST_TIMEOUT,
        stream=True,
    ) as resp:
        resp.raise_for_status()


def probe_host(host: str, timeout: float = 3.05) -> dict[str, Any]:
    """Probe one host for an IGX device via admin/version."""
    try:
        resp = requests.get(
            io_url(host, "admin/version", "value.json"),
            timeout=(timeout, timeout),
        )
        if resp.status_code != 200:
            return {}
        info: dict[str, Any] = {"host": host.split(":")[0], "version": resp.json()}
        try:
            info["device_type"] = get_json(host, "admin/device_type", "value.json")
        except Exception:
            pass
        return info
    except Exception:
        return {}
