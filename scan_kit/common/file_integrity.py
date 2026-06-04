"""Pyramid ``file_integrity_checker`` sidecars (``.md5``) used by map2map / DCS.

Sidecar layout (64-bit little-endian, 52 bytes total) matches
``ptc_core`` ``file_integrity_checker::commit_file_integrity``:

- ``size_t`` hex digest length (always 32)
- ``int`` random salt
- 32 ASCII hex chars (MD5 of file bytes + decimal salt string)
- ``time_t`` file mtime via ``mktime(gmtime(last_write_time))``
"""

from __future__ import annotations

import hashlib
import os
import secrets
import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

HASH_FILE_EXT = ".md5"
HEX_DIGEST_LEN = 32
_SIDECAR_MIN_SIZE = 8 + 4 + HEX_DIGEST_LEN + 8  # 52 on 64-bit


class IntegrityStatus(IntEnum):
    """Mirrors ``pyramid::file_integrity_checker::file_integrity_ret_code_e``."""

    OK = 0
    SOURCE_FILE_NOT_EXIST = 1
    HASH_FILE_NOT_EXIST = 2
    OPEN_ERR = 3
    READ_ERR = 4
    TIMESTAMP_ERR = 5
    HASH_ERR = 6
    DIR_INCONSISTENT = 7


@dataclass(frozen=True)
class FileIntegritySidecar:
    """Parsed contents of a ``.md5`` sidecar."""

    hex_digest: str
    salt: int
    stored_mtime: int
    hex_length: int = HEX_DIGEST_LEN


@dataclass(frozen=True)
class IntegrityCheckResult:
    status: IntegrityStatus
    sidecar: FileIntegritySidecar | None = None
    expected_digest: str | None = None


def sidecar_path(file_path: str | Path) -> Path:
    """Return the sidecar path (``foo.xml`` → ``foo.xml.md5``)."""
    p = Path(file_path)
    return Path(f"{p}{HASH_FILE_EXT}")


def is_sidecar_path(path: str | Path) -> bool:
    return str(path).endswith(HASH_FILE_EXT)


def source_path_from_sidecar(sidecar_file: str | Path) -> Path:
    """Return the data file for a sidecar (``foo.xml.md5`` → ``foo.xml``)."""
    p = Path(sidecar_file)
    suffix = HASH_FILE_EXT
    name = p.name
    if not name.endswith(suffix):
        raise ValueError(f"not a sidecar path: {p}")
    return p.with_name(name[: -len(suffix)])


def compute_hex_digest(file_data: bytes, salt: int) -> str:
    """MD5(file_data ‖ ascii_decimal(salt)), lowercase hex."""
    digest = hashlib.md5()
    digest.update(file_data)
    digest.update(str(salt).encode("ascii"))
    return digest.hexdigest()


def pyramid_utc_mtime(file_path: str | Path) -> int:
    """Match C++ ``mktime(gmtime(last_write_time))`` used at commit/check."""
    mtime = os.path.getmtime(file_path)
    return int(time.mktime(time.gmtime(mtime)))


def parse_sidecar(sidecar_path: str | Path) -> FileIntegritySidecar:
    """Read and parse a binary ``.md5`` sidecar."""
    data = Path(sidecar_path).read_bytes()
    if len(data) < _SIDECAR_MIN_SIZE:
        raise ValueError(f"sidecar too short ({len(data)} bytes): {sidecar_path}")

    hex_length = struct.unpack_from("<Q", data, 0)[0]
    if hex_length > 1024:
        raise ValueError(f"implausible digest length {hex_length}")

    salt = struct.unpack_from("<i", data, 8)[0]
    hex_start = 12
    hex_end = hex_start + hex_length
    if len(data) < hex_end + 8:
        raise ValueError(f"sidecar truncated: {sidecar_path}")

    hex_digest = data[hex_start:hex_end].decode("ascii")
    (stored_mtime,) = struct.unpack_from("<q", data, hex_end)

    return FileIntegritySidecar(
        hex_digest=hex_digest,
        salt=salt,
        stored_mtime=stored_mtime,
        hex_length=hex_length,
    )


def pack_sidecar(hex_digest: str, salt: int, stored_mtime: int) -> bytes:
    """Serialize a sidecar blob."""
    if len(hex_digest) != HEX_DIGEST_LEN:
        raise ValueError(f"digest must be {HEX_DIGEST_LEN} hex chars, got {len(hex_digest)}")
    return (
        struct.pack("<QI", HEX_DIGEST_LEN, salt)
        + hex_digest.encode("ascii")
        + struct.pack("<q", stored_mtime)
    )


def verify_file_integrity(
    file_path: str | Path,
    *,
    check_mtime: bool = True,
    mtime_tolerance: float = 1e-6,
) -> IntegrityCheckResult:
    """Verify ``file_path`` against its ``.md5`` sidecar."""
    path = Path(file_path)
    if not path.is_file():
        return IntegrityCheckResult(IntegrityStatus.SOURCE_FILE_NOT_EXIST)

    md5_path = sidecar_path(path)
    if not md5_path.is_file():
        return IntegrityCheckResult(IntegrityStatus.HASH_FILE_NOT_EXIST)

    try:
        sidecar = parse_sidecar(md5_path)
    except (OSError, ValueError):
        return IntegrityCheckResult(IntegrityStatus.READ_ERR)

    file_data = path.read_bytes()
    expected = compute_hex_digest(file_data, sidecar.salt)
    if expected != sidecar.hex_digest:
        return IntegrityCheckResult(
            IntegrityStatus.HASH_ERR,
            sidecar=sidecar,
            expected_digest=expected,
        )

    if check_mtime:
        actual_mtime = pyramid_utc_mtime(path)
        if abs(actual_mtime - sidecar.stored_mtime) > mtime_tolerance:
            return IntegrityCheckResult(IntegrityStatus.TIMESTAMP_ERR, sidecar=sidecar)

    return IntegrityCheckResult(IntegrityStatus.OK, sidecar=sidecar)


def commit_file_integrity(
    file_path: str | Path,
    *,
    salt: int | None = None,
) -> Path:
    """Write a new ``.md5`` sidecar for the current contents of ``file_path``."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if salt is None:
        # boost::uniform_int_distribution<int>(0, numeric_limits<int>::max())
        salt = secrets.randbelow(2**31)

    file_data = path.read_bytes()
    hex_digest = compute_hex_digest(file_data, salt)
    stored_mtime = pyramid_utc_mtime(path)
    md5_path = sidecar_path(path)
    md5_path.write_bytes(pack_sidecar(hex_digest, salt, stored_mtime))
    return md5_path


_STATUS_LABELS: dict[IntegrityStatus, str] = {
    IntegrityStatus.OK: "OK — sidecar matches file",
    IntegrityStatus.SOURCE_FILE_NOT_EXIST: "Data file missing",
    IntegrityStatus.HASH_FILE_NOT_EXIST: "No .md5 sidecar",
    IntegrityStatus.OPEN_ERR: "Open error",
    IntegrityStatus.READ_ERR: "Sidecar unreadable or corrupt",
    IntegrityStatus.TIMESTAMP_ERR: "Timestamp mismatch",
    IntegrityStatus.HASH_ERR: "Digest mismatch",
    IntegrityStatus.DIR_INCONSISTENT: "Directory inconsistent",
}


def status_label(status: IntegrityStatus) -> str:
    return _STATUS_LABELS.get(status, status.name)


def format_mtime(epoch: int) -> str:
    """Display a Pyramid ``time_t`` value stored in the sidecar."""
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(epoch))
