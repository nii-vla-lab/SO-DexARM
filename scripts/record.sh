#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ $# -lt 1 ]]; then
    echo "Usage: TASK='<TASK_DESCRIPTION>' CAMERAS='<CAMERA_CONFIG_OR_none>' $0 <huggingface-user/dataset> [SO-DexARM record options]" >&2
    exit 2
fi

DATASET_REPO_ID=$1
shift
if [[ -z ${TASK:-} ]]; then
    echo "TASK must be set to the dataset task description." >&2
    exit 2
fi
if [[ -z ${CAMERAS:-} ]]; then
    echo "CAMERAS must be set to a draccus camera dictionary, or to 'none' for state-only recording." >&2
    exit 2
fi
NUM_EPISODES=${NUM_EPISODES:-10}
EPISODE_TIME_S=${EPISODE_TIME_S:-30}
RESET_TIME_S=${RESET_TIME_S:-5}
FPS=${FPS:-30}

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m lerobot.scripts.lerobot_so_dexarm record \
    --repo-id "$DATASET_REPO_ID" \
    --single-task "$TASK" \
    --cameras "$CAMERAS" \
    --num-episodes "$NUM_EPISODES" \
    --episode-time-s "$EPISODE_TIME_S" \
    --reset-time-s "$RESET_TIME_S" \
    --fps "$FPS" \
    "$@"
