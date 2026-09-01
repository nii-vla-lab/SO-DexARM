#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ $# -lt 1 ]]; then
    echo "Usage: TASK='<EXACT_TRAINING_INSTRUCTION>' CAMERAS='<CAMERA_CONFIG_OR_none>' $0 <pretrained-policy-path> [SO-DexARM eval options]" >&2
    exit 2
fi
if [[ -z ${TASK:-} ]]; then
    echo "TASK must be set to the exact language instruction used for SmolVLA training." >&2
    exit 2
fi
if [[ -z ${CAMERAS:-} ]]; then
    echo "CAMERAS must match the policy's training camera configuration, or be 'none' for state-only." >&2
    exit 2
fi

POLICY_PATH=$1
shift
EVAL_REPO_ID=${EVAL_REPO_ID:-local/eval_so-dexarm-smolvla}
NUM_EPISODES=${NUM_EPISODES:-5}
EPISODE_TIME_S=${EPISODE_TIME_S:-30}
RESET_TIME_S=${RESET_TIME_S:-10}
FPS=${FPS:-30}
MAX_RELATIVE_TARGET=${MAX_RELATIVE_TARGET:-2.0}

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m lerobot.scripts.lerobot_so_dexarm eval \
    --pretrained-policy-path "$POLICY_PATH" \
    --repo-id "$EVAL_REPO_ID" \
    --task "$TASK" \
    --cameras "$CAMERAS" \
    --num-episodes "$NUM_EPISODES" \
    --episode-time-s "$EPISODE_TIME_S" \
    --reset-time-s "$RESET_TIME_S" \
    --fps "$FPS" \
    --max-relative-target "$MAX_RELATIVE_TARGET" \
    "$@"
