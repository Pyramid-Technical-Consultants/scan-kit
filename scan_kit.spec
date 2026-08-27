# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building scan-kit as a single executable.

Usage:
    pyinstaller scan_kit.spec          # one-dir (faster builds, for testing)
    pyinstaller scan_kit.spec --onefile # single exe (for distribution)
"""

import ctypes.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None
ROOT = Path(SPECPATH)


def _drop_test_tree_datas(datas: list) -> list:
    return [
        (src, dest)
        for src, dest in datas
        if "/tests/" not in src.replace("\\", "/")
    ]


# ---------------------------------------------------------------------------
# On Linux, sounddevice depends on the system libportaudio which PyInstaller
# won't bundle automatically.  Find it and add it as an extra binary.
# ---------------------------------------------------------------------------
_extra_binaries = []
if sys.platform.startswith("linux"):
    _pa = ctypes.util.find_library("portaudio")
    if _pa:
        import subprocess
        _ldconfig = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True,
        )
        for line in _ldconfig.stdout.splitlines():
            if "libportaudio" in line and "=>" in line:
                _so_path = line.split("=>")[-1].strip()
                _extra_binaries.append((_so_path, "."))
                break

# ---------------------------------------------------------------------------
# Third-party packages with non-Python runtime assets
# ---------------------------------------------------------------------------
_pyside6_datas, _pyside6_binaries, _pyside6_hiddenimports = collect_all("PySide6")

# vispy loads GLSL shaders from disk at import time; hiddenimports alone are not enough.
_vispy_datas, _vispy_binaries, _vispy_hiddenimports = collect_all("vispy")
_vispy_datas = _drop_test_tree_datas(_vispy_datas)

_mpl_datas = collect_data_files("matplotlib", includes=["**/mpl-data/**"])

_assets_dir = ROOT / "scan_kit" / "assets"
_app_datas = [(str(_assets_dir), "scan_kit/assets")]

# ---------------------------------------------------------------------------
# scan_kit submodules imported dynamically (--run-view, warm worker, registry).
# ---------------------------------------------------------------------------
_scan_kit_hiddenimports = collect_submodules("scan_kit.data")
_scan_kit_hiddenimports += collect_submodules("scan_kit.views")
_scan_kit_hiddenimports += collect_submodules("scan_kit.workflows")
_scan_kit_hiddenimports += collect_submodules("scan_kit.common")
_scan_kit_hiddenimports += collect_submodules("scan_kit.igx")

hiddenimports = list(
    dict.fromkeys(
        [
            "scan_kit.app",
            "scan_kit.qt_launcher",
            "scan_kit.common.view_runner",
            "scan_kit.common.matplotlib_backend",
            # third-party modules that PyInstaller sometimes misses
            "scipy.signal",
            "scipy.fft",
            "scipy.fft._pocketfft",
            "sounddevice",
            "matplotlib.backends.backend_qtagg",
            "matplotlib.backends.backend_qt",
            "matplotlib.backends.backend_tkagg",
            "tkinter",
            "PySide6.QtSvg",
            "websocket",
            "msgpack",
            "requests",
            *_scan_kit_hiddenimports,
            *_pyside6_hiddenimports,
            *_vispy_hiddenimports,
        ],
    ),
)

a = Analysis(
    [str(ROOT / "scan_kit" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=_extra_binaries + _pyside6_binaries + _vispy_binaries,
    datas=_pyside6_datas + _app_datas + _vispy_datas + _mpl_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        "matplotlib": {
            "backends": ["QtAgg", "TkAgg"],
        },
    },
    runtime_hooks=[str(ROOT / "scan_kit" / "pyi_rth_frozen.py")],
    excludes=[
        "test_data",
        "pytest",
        "IPython",
        "notebook",
        "sphinx",
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# --onefile build (default: produces a single executable)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="scan-kit",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(_assets_dir / "icon.ico") if sys.platform == "win32" else None,
)
