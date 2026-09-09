"""Linux frozen-build environment helpers (no Qt imports)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_APP_DIR_ENV = "SCAN_KIT_APP_DIR"


def frozen_app_dir() -> Path | None:
    """Directory containing the frozen ``scan-kit`` binary, when applicable."""
    if sys.platform != "linux" or not getattr(sys, "frozen", False):
        return None
    raw = os.environ.get(_APP_DIR_ENV)
    if raw:
        return Path(raw)
    return Path(sys.executable).resolve().parent


def _ensure_stdio() -> None:
    """GUI launches from file managers may provide no usable stderr/stdout."""
    for fd in (0, 1, 2):
        try:
            os.fstat(fd)
        except OSError:
            flags = os.O_RDONLY if fd == 0 else os.O_WRONLY
            devnull = os.open(os.devnull, flags)
            os.dup2(devnull, fd)
            if fd != 0:
                os.close(devnull)


def _ensure_app_working_directory() -> None:
    app_dir = Path(sys.executable).resolve().parent
    os.environ[_APP_DIR_ENV] = str(app_dir)
    try:
        os.chdir(app_dir)
    except OSError:
        pass


def prepare_linux_frozen_env() -> None:
    """Keep bundled GLib from loading incompatible system GIO/GTK plugins.

    PyInstaller bundles an older ``libglib`` (pulled in via Qt/D-Bus deps).
    GIO then tries to load system modules (gvfs, dconf, ibus) built against a
    newer GLib, which produces undefined-symbol warnings and broken input-method
    plugins.  Call as early as possible in the process.
    """
    if sys.platform != "linux" or not getattr(sys, "frozen", False):
        return

    _ensure_stdio()
    _ensure_app_working_directory()

    # Prevent GIO from scanning /usr/lib/.../gio/modules/.
    for key in (
        "GIO_MODULE_DIR",
        "GIO_MODULE_DIR_GSETTINGS",
        "GIO_MODULE_DIR_NETWORK",
        "GIO_MODULE_DIR_VOLUMEMONITOR",
    ):
        os.environ[key] = ""

    # Empty GTK_IM_MODULE lets GTK fall back to gsettings → often "ibus", which
    # then fails with g_task_set_static_name against bundled GLib. Force a
    # non-ibus module instead.
    os.environ["QT_IM_MODULE"] = "simple"
    os.environ["GTK_IM_MODULE"] = "xim"
    os.environ["XMODIFIERS"] = "@im=none"

    # Avoid atk-bridge signature mismatch noise from accessibility probing.
    os.environ.setdefault("NO_AT_BRIDGE", "1")
    # Don't consult host dconf for gtk-im-module / theme plugins.
    os.environ.setdefault("GSETTINGS_BACKEND", "memory")
