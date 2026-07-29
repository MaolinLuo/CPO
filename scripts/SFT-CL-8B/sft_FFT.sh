#!/bin/bash
set -euo pipefail

BASE_MODEL=$1
BASE_PATH=$2
TASK_ID=$3
shift 3
DATASETS=("$@")

export PYTHONPATH=src:${PYTHONPATH:-}

GLOBAL_BATCH_SIZE=8
BATCH_PER_DEVICE=2
# GLOBAL_BATCH_SIZE=8
# BATCH_PER_DEVICE=1
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -s -d,)
    export CUDA_VISIBLE_DEVICES=$ALL_GPUS
fi
IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_DEVICES=${#GPULIST[@]}
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

CUR_DATASET=${DATASETS[$((TASK_ID - 1))]}

if [[ "$CUR_DATASET" == "MedBookVQA" ]]; then
    NUM_TRAIN_EPOCHS=2
elif [[ "$CUR_DATASET" == "Navigation" ]]; then
    NUM_TRAIN_EPOCHS=3
else
    NUM_TRAIN_EPOCHS=1
fi

if [ $TASK_ID -eq 1 ]; then
    MODEL_NAME=$BASE_MODEL
else
    PREV_DATASET=${DATASETS[$((TASK_ID - 2))]}
    MODEL_NAME=./checkpoints/Qwen3-VL-8B/FFT-CL/model/${PREV_DATASET}
fi

echo "======================================"
echo "Training on dataset: $CUR_DATASET"
echo "Task ID: $TASK_ID"
echo "Using model: $MODEL_NAME"
echo "======================================"

DATA_PATH=${BASE_PATH}/${CUR_DATASET}/jsons/train/data.json
MEDIA_DIR=${BASE_PATH}/${CUR_DATASET}/images
OUTPUT_DIR=./checkpoints/Qwen3-VL-8B/FFT-CL/model/${CUR_DATASET}
mkdir -p ${OUTPUT_DIR}
    
# If you want to set the min pixels and max pixels for Qwen3-VL, You should set as (N * 32 * 32)

deepspeed src/train/train_sft.py \
    --use_liger_kernel False \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --image_folder $MEDIA_DIR \
    --prompt_path src/dataset/prompts_1.yaml \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((128 * 32 * 32)) \
    --image_max_pixels $((512 * 32 * 32)) \
    --learning_rate 5e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing False \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4
