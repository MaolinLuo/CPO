#!/bin/bash
set -euo pipefail

BASE_MODEL=$1
BASE_PATH=$2
TASK_ID=$3
shift 3
DATASETS=("$@")

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src:${PYTHONPATH:-}

NUM_GENERATIONS=8
NUM_DEVICES=$(nvidia-smi --list-gpus | wc -l)   # 8
# per_device_train_batch_size means number of generations per device, and not the number of prompts per device 
BATCH_PER_DEVICE=8   # https://github.com/huggingface/trl/issues/3061
MAX_STEPS=600
LR=5e-6

CUR_DATASET=${DATASETS[$((TASK_ID - 1))]}
if [ "$CUR_DATASET" == "MedBookVQA" ] || [ "$CUR_DATASET" == "CVQA" ] || [ "$CUR_DATASET" == "Chemistry" ]; then
    MAX_STEPS=300
elif [ "$CUR_DATASET" == "InstructFollow" ]; then
    MAX_STEPS=100
fi

if [ $TASK_ID -eq 1 ]; then
    MODEL_NAME=$BASE_MODEL
else
    PREV_DATASET=${DATASETS[$((TASK_ID - 2))]}
    MODEL_NAME=./checkpoints/Qwen3-VL-4B/GRPO-CL/training/${PREV_DATASET}
fi

echo "======================================"
echo "Training on dataset: $CUR_DATASET"
echo "Task ID: $TASK_ID"
echo "Using model: $MODEL_NAME"
echo "======================================"

DATA_PATH=${BASE_PATH}/${CUR_DATASET}/jsons/train/data.json
MEDIA_DIR=${BASE_PATH}/${CUR_DATASET}/images
OUTPUT_DIR=./checkpoints/Qwen3-VL-4B/GRPO-CL/training/${CUR_DATASET}
MERGE_SAVE_PATH=./checkpoints/Qwen3-VL-4B/GRPO-CL/model/${CUR_DATASET}

mkdir -p ${OUTPUT_DIR}
    
deepspeed src/train/train_grpo.py \
    --use_liger_kernel False \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --image_folder $MEDIA_DIR \
    --prompt_path src/dataset/prompts_2.yaml \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --max_steps $MAX_STEPS \
    --num_generations $NUM_GENERATIONS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps 4 \
    --max_completion_length 2048 \
    --image_min_pixels $((128 * 32 * 32)) \
    --image_max_pixels $((512 * 32 * 32)) \
    --learning_rate $LR \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 30 \
    --save_total_limit 20 \
    --dataloader_num_workers 4 \
    --beta 0.0 \
    --temperature 0.8 \
    --top_p 1.0 \
    --top_k 0 \
    --repetition_penalty 1.0 \
    --steps_per_generation 16 \
    --epsilon 0.2 \
    --epsilon_high 0.27

