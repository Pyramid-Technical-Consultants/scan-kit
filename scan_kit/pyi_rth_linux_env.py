"""PyInstaller runtime hook: Linux GLib/GIO isolation (runs before app imports)."""

import sys

if sys.platform.startswith("linux") and getattr(sys, "frozen", False):
    from scan_kit.common.linux_frozen_env import prepare_linux_frozen_env

    prepare_linux_frozen_env()
