#!/bin/bash
set -euo pipefail

BASE_MODEL=$1
BASE_PATH=$2
TASK_ID=$3
DATA_RATIO=$4
SVD_MODE=$5
shift 5
DATASETS=("$@")

export PYTHONPATH=src:${PYTHONPATH:-}

NUM_DEVICES=$(nvidia-smi --list-gpus | wc -l)
CUR_DATASET=${DATASETS[$((TASK_ID - 1))]}

if [ $TASK_ID -eq 1 ]; then
    MODEL_NAME=$BASE_MODEL
    SPACE_PATH=./checkpoints/Qwen3-VL-8B/KeepLoRA-CL/lora_gradients/Init
else
    PREV_DATASET=${DATASETS[$((TASK_ID - 2))]}
    MODEL_NAME=./checkpoints/Qwen3-VL-8B/KeepLoRA-CL/model/${PREV_DATASET}
    SPACE_PATH=./checkpoints/Qwen3-VL-8B/KeepLoRA-CL/lora_gradients/${PREV_DATASET}
fi

if [ "${SVD_MODE}" = "energy" ]; then
    MODEL_NAME=./checkpoints/Qwen3-VL-8B/KeepLoRA-CL/model/${CUR_DATASET}
fi

DATA_PATH=${BASE_PATH}/${CUR_DATASET}/jsons/train/data.json
MEDIA_DIR=${BASE_PATH}/${CUR_DATASET}/images
OUTPUT_DIR=./checkpoints/Qwen3-VL-8B/KeepLoRA-CL/lora_gradients/${CUR_DATASET}
RANK=128
ENERGY_THRESHOLD=0.995


gpu_list=""
for ((i=0; i<NUM_DEVICES; i++)); do
    gpu_list+="$i,"
done
gpu_list=${gpu_list%,}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$gpu_list}"

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python src/train/extract_gradients.py \
        --data_path $DATA_PATH \
        --image_folder $MEDIA_DIR \
        --prompt_path src/dataset/prompts_1.yaml \
        --image_min_pixels $((128 * 32 * 32)) \
        --image_max_pixels $((512 * 32 * 32)) \
        --model_id $MODEL_NAME \
        --rank $RANK \
        --energy_threshold $ENERGY_THRESHOLD \
        --space_path $SPACE_PATH \
        --output_dir $OUTPUT_DIR \
        --data_ratio $DATA_RATIO \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --disable_flash_attn2 False &
done

wait

python src/train/extract_gradients.py \
    --model_id $MODEL_NAME \
    --rank $RANK \
    --energy_threshold $ENERGY_THRESHOLD \
    --space_path $SPACE_PATH \
    --output_dir $OUTPUT_DIR \
    --num_chunks $CHUNKS \
    --merge_only \
    --svd_mode $SVD_MODE \
    --disable_flash_attn2 False \
