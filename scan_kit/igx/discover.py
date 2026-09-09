"""Discover IGX devices on a local subnet."""

from __future__ import annotations

import ipaddress
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .http import probe_host


def discover_subnet(
    subnet: str,
    *,
    port: int = 80,
    timeout: float = 1.0,
    max_workers: int = 64,
) -> list[dict[str, Any]]:
    """Scan an IPv4 subnet (CIDR) for reachable IGX devices."""
    net = ipaddress.ip_network(subnet, strict=False)
    hosts = [str(ip) for ip in net.hosts()]
    if len(hosts) > 1024:
        raise ValueError(f"subnet too large ({len(hosts)} hosts); use a /22 or smaller")

    found: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(probe_host, f"{ip}:{port}" if port != 80 else ip, timeout): ip
            for ip in hosts
        }
        for fut in as_completed(futures):
            info = fut.result()
            if info:
                found.append(info)
    return sorted(found, key=lambda d: d.get("host", ""))
