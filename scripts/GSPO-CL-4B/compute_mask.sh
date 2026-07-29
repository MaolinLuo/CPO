#!/bin/bash
set -euo pipefail

BASE_MODEL=$1
TASK_ID=$2
shift 2
DATASETS=("$@")

export PYTHONPATH=src:${PYTHONPATH:-}

echo "======================================"
echo "Computing importance mask for Task ${TASK_ID}: ${DATASETS[$((TASK_ID - 1))]}"
echo "======================================"

python src/compute_importance_mask.py \
    --base_model "$BASE_MODEL" \
    --task_id "$TASK_ID" \
    --datasets "${DATASETS[@]}" \
    --top_percent 10.0 \
    --checkpoint_dir ./checkpoints/Qwen3-VL-4B/GSPO-CL

echo "Mask computation done."
