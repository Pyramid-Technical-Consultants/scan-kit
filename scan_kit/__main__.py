"""Allow running scan-kit with ``python -m scan_kit``.

Also serves as the PyInstaller entry point so that ``scan_kit.app``
is imported as part of its package (enabling relative imports).
"""

from scan_kit.app import main

if __name__ == "__main__":
    main()
