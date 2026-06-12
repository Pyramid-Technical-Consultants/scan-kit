"""PyInstaller runtime hooks for frozen scan-kit builds (run before app imports)."""

import os
import sys

if getattr(sys, "frozen", False):
    # Matplotlib views must use the bundled PySide6 stack, not TkAgg/Gtk.
    os.environ.setdefault("MPLBACKEND", "QtAgg")

if sys.platform.startswith("linux") and getattr(sys, "frozen", False):
    from scan_kit.common.linux_frozen_env import prepare_linux_frozen_env

    prepare_linux_frozen_env()
