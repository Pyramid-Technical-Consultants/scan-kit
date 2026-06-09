# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building scan-kit as a single executable.

Usage:
    pyinstaller scan_kit.spec          # one-dir (faster builds, for testing)
    pyinstaller scan_kit.spec --onefile # single exe (for distribution)
"""

import ctypes.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

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
# PySide6: plugins and Qt resource files
# ---------------------------------------------------------------------------
_pyside6_datas, _pyside6_binaries, _pyside6_hiddenimports = collect_all("PySide6")

# ---------------------------------------------------------------------------
# Collect all scan_kit submodules so PyInstaller bundles them even though
# some are only imported dynamically (e.g. via --run-view).
# ---------------------------------------------------------------------------
hiddenimports = [
    "scan_kit.qt_launcher",
    "scan_kit.common.plot_colors",
    # view modules (dynamically imported via importlib in --run-view / warm-worker mode)
    "scan_kit.views.position_error_energy",
    "scan_kit.views.position_error_distribution_timeslice",
    "scan_kit.views.position_error_distribution_spot",
    "scan_kit.views.position_error_outliers_spot",
    "scan_kit.views.beam_motion_energy",
    "scan_kit.views.session_log_compare",
    "scan_kit.views.ic_beam_trajectory",
    "scan_kit.views.position_scatter",
    "scan_kit.views.dose_ratios_energy",
    "scan_kit.views.dose_ratios_position",
    "scan_kit.views.dose_ratios_spot_time",
    "scan_kit.views.current_ratios",
    "scan_kit.views.dose_error_energy",
    "scan_kit.views.dose_error_energy_mean",
    "scan_kit.views.dose_error_mu",
    "scan_kit.views.spot_delivery_time",
    "scan_kit.views.sigma_energy",
    "scan_kit.views.beam_off_rampdown",
    "scan_kit.views.beam_on_off_current",
    "scan_kit.views.ic_timeslice_replay",
    "scan_kit.views.ic_timeslice_replay_derived",
    "scan_kit.views.field_timeslice_replay",
    "scan_kit.views.timeslice_replay_common",
    "scan_kit.views.timeslice_replay_ui",
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
    "scan_kit.common.position_error_distribution",
    "scan_kit.common.view_runner",
    # third-party modules that PyInstaller sometimes misses
    "scipy.signal",
    "scipy.fft",
    "scipy.fft._pocketfft",
    "sounddevice",
    "matplotlib.backends.backend_tkagg",
    "tkinter",
    *_pyside6_hiddenimports,
]

a = Analysis(
    [str(ROOT / "scan_kit" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=_extra_binaries + _pyside6_binaries,
    datas=_pyside6_datas,
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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
