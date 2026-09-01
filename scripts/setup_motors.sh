#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ $# -lt 1 || ( "$1" != "right" && "$1" != "left" ) ]]; then
    echo "Usage: $0 <right|left> [setup-motors options]" >&2
    echo "Only one physical motor must be connected for each interactive prompt." >&2
    exit 2
fi

SIDE=$1
shift
cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m lerobot.scripts.lerobot_so_dexarm setup-motors --side "$SIDE" "$@"
