#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
FPS=${FPS:-30}
SIDES=${SIDES:-both}
TELEOP_MODE=${TELEOP_MODE:-arm-and-hand}

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m lerobot.scripts.lerobot_so_dexarm teleop \
    --fps "$FPS" \
    --sides "$SIDES" \
    --teleop-mode "$TELEOP_MODE" \
    "$@"
