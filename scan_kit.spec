# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building scan-kit as a single executable.

Usage:
    pyinstaller scan_kit.spec          # one-dir (faster builds, for testing)
    pyinstaller scan_kit.spec --onefile # single exe (for distribution)
"""

import ctypes.util
import importlib
import pkgutil
import sys
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH)

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
# Collect all rich._unicode_data submodules (loaded dynamically by rich at
# runtime based on the Unicode version — PyInstaller can't detect these).
# ---------------------------------------------------------------------------
_rich_unicode = importlib.import_module("rich._unicode_data")
_rich_unicode_imports = [
    f"rich._unicode_data.{m.name}"
    for m in pkgutil.iter_modules(_rich_unicode.__path__)
]

# ---------------------------------------------------------------------------
# Collect all scan_kit submodules so PyInstaller bundles them even though
# some are only imported dynamically (e.g. via --run-view).
# ---------------------------------------------------------------------------
hiddenimports = [
    # view modules (dynamically imported via importlib in --run-view mode)
    "scan_kit.views.ic1_position_bars",
    "scan_kit.views.ic1_ic2_error_scatter",
    "scan_kit.views.ic1_ic2_spot_scatter",
    "scan_kit.views.dose_ratios",
    "scan_kit.views.dose_ratios_position",
    "scan_kit.views.dose_ratios_time",
    "scan_kit.views.dose_error_vs_target",
    "scan_kit.views.dose_error_vs_target_mean_scatter",
    "scan_kit.views.spot_delivery_time",
    "scan_kit.views.sigma_boxplots",
    "scan_kit.views.beam_off_rampdown",
    "scan_kit.views.beam_on_off_current",
    "scan_kit.views.ic_timeslice_replay",
    "scan_kit.views.dose_accumulation",
    "scan_kit.views.ic_fft_analysis",
    "scan_kit.views.ic_audio_export",
    # common submodules
    "scan_kit.common.session_meta",
    "scan_kit.common.session_notes",
    "scan_kit.common.session_source",
    "scan_kit.common.sessions",
    "scan_kit.common.schema",
    "scan_kit.common.transform",
    "scan_kit.common.validation",
    "scan_kit.common.processing",
    "scan_kit.common.plotting",
    # third-party modules that PyInstaller sometimes misses
    "scipy.signal",
    "scipy.fft",
    "scipy.fft._pocketfft",
    "sounddevice",
    "matplotlib.backends.backend_tkagg",
    "tkinter",
    # rich unicode data (dynamically loaded based on Unicode version)
    *_rich_unicode_imports,
]

a = Analysis(
    [str(ROOT / "scan_kit" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=_extra_binaries,
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
