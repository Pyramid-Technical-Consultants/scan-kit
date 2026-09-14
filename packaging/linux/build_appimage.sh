#!/usr/bin/env bash
# Assemble a PyInstaller onedir build into an AppImage.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ONEDIR="${ROOT}/dist/scan-kit"
APPDIR="${ROOT}/dist/ScanKit.AppDir"
APPIMAGETOOL="${APPIMAGETOOL:-/tmp/appimagetool}"
APPIMAGETOOL_URL="${APPIMAGETOOL_URL:-https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage}"

if [[ ! -x "${ONEDIR}/scan-kit" ]]; then
  echo "Missing PyInstaller onedir build at ${ONEDIR}/scan-kit" >&2
  echo "Run: python build.py --onedir" >&2
  exit 1
fi

VERSION="$(python -c "from scan_kit import __version__; print(__version__)")"
OUTPUT_NAME="${1:-scan-kit-linux-amd64-${VERSION}.AppImage}"
OUTPUT_PATH="${ROOT}/dist/${OUTPUT_NAME}"

rm -rf "${APPDIR}"
mkdir -p "${APPDIR}/usr/scan-kit"
cp -a "${ONEDIR}/." "${APPDIR}/usr/scan-kit/"
chmod +x "${APPDIR}/usr/scan-kit/scan-kit"

cp "${ROOT}/packaging/linux/AppRun" "${APPDIR}/AppRun"
chmod +x "${APPDIR}/AppRun"

cp "${ROOT}/packaging/linux/appimage.desktop" "${APPDIR}/scan-kit.desktop"
mkdir -p "${APPDIR}/usr/share/applications" "${APPDIR}/usr/share/metainfo"
cp "${ROOT}/packaging/linux/appimage.desktop" "${APPDIR}/usr/share/applications/scan-kit.desktop"
cp "${ROOT}/packaging/linux/scan-kit.appdata.xml" "${APPDIR}/usr/share/metainfo/scan-kit.appdata.xml"
ln -sf "usr/scan-kit/scan-kit" "${APPDIR}/scan-kit"

ICON_PNG="${ROOT}/scan_kit/assets/icon.png"
SCAN_KIT_APPDIR="${APPDIR}" SCAN_KIT_ICON="${ICON_PNG}" python - <<'PY'
import os
from pathlib import Path

from scan_kit.common.linux_desktop import install_appdir_icons

install_appdir_icons(Path(os.environ["SCAN_KIT_APPDIR"]), Path(os.environ["SCAN_KIT_ICON"]))
PY

if [[ ! -x "${APPIMAGETOOL}" ]]; then
  echo "Downloading appimagetool…"
  wget -q "${APPIMAGETOOL_URL}" -O "${APPIMAGETOOL}"
  chmod +x "${APPIMAGETOOL}"
fi

rm -f "${OUTPUT_PATH}"
# GitHub Actions runners often lack FUSE; extract-and-run avoids that requirement.
"${APPIMAGETOOL}" --appimage-extract-and-run "${APPDIR}" "${OUTPUT_PATH}"
chmod +x "${OUTPUT_PATH}"

echo "AppImage ready: ${OUTPUT_PATH}"
