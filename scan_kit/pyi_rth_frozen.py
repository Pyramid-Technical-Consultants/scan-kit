"""PyInstaller runtime hooks for frozen scan-kit builds (run before app imports)."""

import sys

if sys.platform.startswith("linux") and getattr(sys, "frozen", False):
    from scan_kit.common.linux_desktop import ensure_linux_desktop_integration
    from scan_kit.common.linux_frozen_env import prepare_linux_frozen_env

    prepare_linux_frozen_env()
    ensure_linux_desktop_integration()
