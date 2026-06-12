"""Linux frozen-build environment helpers (no Qt imports)."""

from __future__ import annotations

import os
import sys


def prepare_linux_frozen_env() -> None:
    """Keep bundled GLib from loading incompatible system GIO/GTK plugins.

    PyInstaller bundles an older ``libglib`` (pulled in via Qt/D-Bus deps).
    GIO then tries to load system modules (gvfs, dconf, ibus) built against a
    newer GLib, which produces undefined-symbol warnings and broken input-method
    plugins.  Call as early as possible in the process.
    """
    if sys.platform != "linux" or not getattr(sys, "frozen", False):
        return

    # Prevent GIO from scanning /usr/lib/.../gio/modules/.
    for key in (
        "GIO_MODULE_DIR",
        "GIO_MODULE_DIR_GSETTINGS",
        "GIO_MODULE_DIR_NETWORK",
        "GIO_MODULE_DIR_VOLUMEMONITOR",
    ):
        os.environ[key] = ""

    # IBus/GTK IM modules link against system GLib and fail with bundled GLib.
    os.environ["QT_IM_MODULE"] = "simple"
    os.environ["GTK_IM_MODULE"] = ""

    # Avoid atk-bridge signature mismatch noise from accessibility probing.
    os.environ.setdefault("NO_AT_BRIDGE", "1")
