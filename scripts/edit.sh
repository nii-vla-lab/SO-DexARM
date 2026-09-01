#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

cd "$REPO_ROOT"
if [[ $# -eq 0 ]]; then
    exec "$PYTHON_BIN" -m lerobot.scripts.lerobot_edit_dataset --help
fi
exec "$PYTHON_BIN" -m lerobot.scripts.lerobot_edit_dataset "$@"
