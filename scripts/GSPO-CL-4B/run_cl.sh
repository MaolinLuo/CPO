#!/bin/bash
set -euo pipefail

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

MAIN_LOG=./checkpoints/Qwen3-VL-4B/GSPO-CL/training.log
mkdir -p ./checkpoints/Qwen3-VL-4B/GSPO-CL
exec > >(tee "$MAIN_LOG") 2>&1

echo "======================================"
echo "Starting GSPO Continual Learning Training"
echo "======================================"

BASE_MODEL=/your_model_path/Qwen/Qwen3-VL-4B-Instruct
BASE_PATH=/your_data_path

DATASETS=(
    "MedBookVQA"
    "Navigation"
    "We-Math2"
    "Puzzle"
    "FinMME"
)

for TASK_ID in $(seq 1 ${#DATASETS[@]})
do
    conda activate trlQwen
    bash scripts/GSPO-CL-4B/gspo.sh "$BASE_MODEL" "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}"
    conda activate vllmQwen
    bash scripts/GSPO-CL-4B/eval.sh "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}"
done

echo ""
echo "======================================"
echo "All ${#DATASETS[@]} tasks finished!"
echo "======================================"
