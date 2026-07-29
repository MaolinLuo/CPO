#!/bin/bash
set -euo pipefail

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

MAIN_LOG=./checkpoints/Qwen3-VL-2B/KeepLoRA-CL/training.log
mkdir -p ./checkpoints/Qwen3-VL-2B/KeepLoRA-CL
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "======================================"
echo "Starting Continual Learning Training"     
echo "======================================"

BASE_MODEL=/your_model_path/Qwen/Qwen3-VL-2B-Instruct
BASE_PATH=/your_data_path

DATASETS=(
    "MedBookVQA"
    "Navigation"
    "We-Math2"
    "Puzzle"
    "FinMME" 
)

ENERGY_THRESHOLD=0.6
conda activate trlQwen
bash scripts/KeepLoRA-CL/extract_weights.sh "$BASE_MODEL" "$ENERGY_THRESHOLD"
DATA_RATIO=0.2
for TASK_ID in $(seq 1 ${#DATASETS[@]})
do
    conda activate trlQwen
    bash scripts/KeepLoRA-CL/extract_gradients.sh "$BASE_MODEL" "$BASE_PATH" "$TASK_ID" "$DATA_RATIO" fixed_rank "${DATASETS[@]}"
    bash scripts/KeepLoRA-CL/sft_keeplora.sh "$BASE_MODEL" "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}"
    bash scripts/KeepLoRA-CL/extract_gradients.sh "$BASE_MODEL" "$BASE_PATH" "$TASK_ID" "$DATA_RATIO" energy "${DATASETS[@]}"

    conda activate vllmQwen
    bash scripts/KeepLoRA-CL/eval.sh "$BASE_PATH" "$TASK_ID" "${DATASETS[@]}"
done

echo ""
echo "======================================"
echo "All ${#DATASETS[@]} tasks finished!"
echo "======================================"
