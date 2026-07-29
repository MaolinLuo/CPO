#!/bin/bash
set -euo pipefail

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

MAIN_LOG=./checkpoints/Qwen3-VL-4B/OLoRA-CL/training.log
mkdir -p ./checkpoints/Qwen3-VL-4B/OLoRA-CL
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "======================================"
echo "Starting Continual Learning Training"
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
    bash scripts/OLoRA-CL-4B/sft_olora.sh "$BASE_MODEL" "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}"
    conda activate vllmQwen
    bash scripts/OLoRA-CL-4B/eval.sh "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}"
done

echo ""
echo "======================================"
echo "All ${#DATASETS[@]} tasks finished!"
echo "======================================"
