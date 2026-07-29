#!/bin/bash
set -euo pipefail

BASE_MODEL=$1
ENERGY_THRESHOLD=$2
OUTPUT_DIR=./checkpoints/Qwen3-VL-4B/KeepLoRA-CL/lora_gradients/Init
export PYTHONPATH=src:${PYTHONPATH:-}
python src/train/extract_weights.py \
    --model_id $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --energy_threshold $ENERGY_THRESHOLD \
    --disable_flash_attn2 False \
