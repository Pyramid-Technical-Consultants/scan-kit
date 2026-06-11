"""Windows process identity helpers (no Qt imports)."""

from __future__ import annotations

import ctypes
import sys

_WIN_APP_USER_MODEL_ID = "ProtonCare.ScanKit"


def prepare_windows_app_identity() -> None:
    """Decouple the process from python.exe for Windows taskbar branding.

    Call as early as possible in the process, before ``QApplication`` is
    constructed.
    """
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(  # type: ignore[attr-defined]
            _WIN_APP_USER_MODEL_ID,
        )
    except Exception:
        pass
