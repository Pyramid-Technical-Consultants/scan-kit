#!/usr/bin/env python3
"""Build scan-kit into a single executable for the current platform.

Usage:
    python build.py              # build single executable
    python build.py --clean      # wipe build/ and dist/ first
    python build.py --onedir     # one-directory bundle (faster, for testing)
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "scan_kit.spec"


def _ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("Installing PyInstaller…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller>=6.0"])


def clean() -> None:
    for d in ("build", "dist"):
        p = ROOT / d
        if p.exists():
            print(f"Removing {p}")
            shutil.rmtree(p)


def build(*, onedir: bool = False) -> Path:
    _ensure_pyinstaller()

    cmd: list[str] = [
        sys.executable, "-m", "PyInstaller",
        str(SPEC),
        "--noconfirm",
        "--clean",
    ]

    if onedir:
        cmd.append("--onedir")

    print(f"Building scan-kit ({platform.system()} {platform.machine()})…")
    print(f"  Command: {' '.join(cmd)}\n")
    subprocess.check_call(cmd, cwd=ROOT)

    if onedir:
        out = ROOT / "dist" / "scan-kit"
    else:
        suffix = ".exe" if platform.system() == "Windows" else ""
        out = ROOT / "dist" / f"scan-kit{suffix}"

    print(f"\nBuild complete: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scan-kit executable")
    parser.add_argument("--clean", action="store_true", help="Remove build artifacts first")
    parser.add_argument("--onedir", action="store_true", help="One-directory bundle instead of single file")
    args = parser.parse_args()

    if args.clean:
        clean()

    artifact = build(onedir=args.onedir)

    if artifact.exists():
        if artifact.is_dir():
            sizes = sum(f.stat().st_size for f in artifact.rglob("*") if f.is_file())
        else:
            sizes = artifact.stat().st_size
        mb = sizes / (1024 * 1024)
        print(f"Size: {mb:.1f} MB")
    else:
        print("WARNING: Expected output not found — check build log above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
