"""Qt launcher entry point (thin module so ``import scan_kit.app`` stays cheap)."""

from __future__ import annotations


def main() -> None:
    """Run the scan-kit GUI or dispatch ``--run-view`` / ``--version``."""
    import sys

    if "--version" in sys.argv or "-V" in sys.argv:
        from scan_kit import __version__

        print(f"scan-kit {__version__}")
        return

    from scan_kit.common.win_identity import prepare_windows_app_identity

    prepare_windows_app_identity()

    from scan_kit.qt_launcher import main as _gui_main

    return _gui_main()
