#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <huggingface-user/dataset> [lerobot-train options]" >&2
    exit 2
fi

DATASET_REPO_ID=$1
shift
DATASET_NAME=${DATASET_REPO_ID##*/}
JOB_NAME=${JOB_NAME:-so-dexarm-act-$DATASET_NAME}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/train/$JOB_NAME}
POLICY_DEVICE=${POLICY_DEVICE:-cuda}
STEPS=${STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-16}
SAVE_FREQ=${SAVE_FREQ:-10000}
NUM_WORKERS=${NUM_WORKERS:-4}

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --policy.type=act \
    --policy.device="$POLICY_DEVICE" \
    --policy.push_to_hub=false \
    --steps="$STEPS" \
    --batch_size="$BATCH_SIZE" \
    --num_workers="$NUM_WORKERS" \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
    --output_dir="$OUTPUT_DIR" \
    --job_name="$JOB_NAME" \
    "$@"
