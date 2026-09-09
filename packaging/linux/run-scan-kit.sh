#!/usr/bin/env bash
# Launch scan-kit with a stable working directory (file-manager friendly).
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"
exec "$APP_DIR/scan-kit" "$@"
