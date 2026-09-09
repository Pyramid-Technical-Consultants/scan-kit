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


def build_appimage(*, output_name: str | None = None) -> Path:
    """Build an AppImage from an existing PyInstaller onedir bundle (Linux only)."""
    if platform.system() != "Linux":
        raise SystemExit("AppImage builds require Linux")

    onedir = ROOT / "dist" / "scan-kit"
    if not (onedir / "scan-kit").is_file():
        raise SystemExit(
            "Missing dist/scan-kit/scan-kit — run `python build.py --onedir` first",
        )

    script = ROOT / "packaging" / "linux" / "build_appimage.sh"
    cmd = ["bash", str(script)]
    if output_name:
        cmd.append(output_name)
    subprocess.check_call(cmd, cwd=ROOT)

    if output_name:
        out = ROOT / "dist" / output_name
    else:
        from scan_kit import __version__

        out = ROOT / "dist" / f"scan-kit-linux-amd64-{__version__}.AppImage"
    if not out.is_file():
        raise SystemExit(f"AppImage build failed: expected {out}")
    print(f"\nAppImage complete: {out}")
    return out


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
    parser.add_argument(
        "--appimage",
        metavar="NAME",
        nargs="?",
        const="",
        help="Build an AppImage (Linux only; requires --onedir output in dist/scan-kit)",
    )
    args = parser.parse_args()

    if args.clean:
        clean()

    if args.appimage is not None:
        name = args.appimage or None
        artifact = build_appimage(output_name=name)
    else:
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
