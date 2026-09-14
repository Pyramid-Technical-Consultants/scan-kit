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
cp "${ROOT}/scan_kit/assets/icon.png" "${APPDIR}/scan-kit.png"
ln -sf scan-kit.png "${APPDIR}/.DirIcon"
ln -sf "usr/scan-kit/scan-kit" "${APPDIR}/scan-kit"

# Freedesktop icon theme paths so Icon=scan-kit resolves before first-run install.
ICON_PNG="${ROOT}/scan_kit/assets/icon.png"
ICON_SVG="${ROOT}/scan_kit/assets/scan-kit-icon.svg"
for size in 16 24 32 48 64 128 256; do
  dest="${APPDIR}/usr/share/icons/hicolor/${size}x${size}/apps"
  mkdir -p "${dest}"
  cp "${ICON_PNG}" "${dest}/scan-kit.png"
done
if [[ -f "${ICON_SVG}" ]]; then
  svg_dest="${APPDIR}/usr/share/icons/hicolor/scalable/apps"
  mkdir -p "${svg_dest}"
  cp "${ICON_SVG}" "${svg_dest}/scan-kit.svg"
fi

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
