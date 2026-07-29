#!/bin/bash
set -euo pipefail

BASE_MODEL=$1
BASE_PATH=$2
TASK_ID=$3
shift 3
DATASETS=("$@")

export PYTHONPATH=src:${PYTHONPATH:-}

GLOBAL_BATCH_SIZE=8
BATCH_PER_DEVICE=8
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -s -d,)
    export CUDA_VISIBLE_DEVICES=$ALL_GPUS
fi
IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_DEVICES=${#GPULIST[@]}
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

CUR_DATASET=${DATASETS[$((TASK_ID - 1))]}

if [ $TASK_ID -eq 1 ]; then
    MODEL_NAME=$BASE_MODEL
else
    PREV_DATASET=${DATASETS[$((TASK_ID - 2))]}
    MODEL_NAME=./checkpoints/Qwen3-VL-2B/RegLoRA-CL/model/${PREV_DATASET}
fi

if [[ "$CUR_DATASET" == "MedBookVQA" ]]; then
    NUM_TRAIN_EPOCHS=2
elif [[ "$CUR_DATASET" == "Navigation" ]]; then
    NUM_TRAIN_EPOCHS=3
else
    NUM_TRAIN_EPOCHS=1
fi

echo "======================================"
echo "Training on dataset: $CUR_DATASET"
echo "Task ID: $TASK_ID"
echo "Using model: $MODEL_NAME"
echo "======================================"

DATA_PATH=${BASE_PATH}/${CUR_DATASET}/jsons/train/data.json
MEDIA_DIR=${BASE_PATH}/${CUR_DATASET}/images
OUTPUT_DIR=./checkpoints/Qwen3-VL-2B/RegLoRA-CL/training/${CUR_DATASET}
MERGE_SAVE_PATH=./checkpoints/Qwen3-VL-2B/RegLoRA-CL/model/${CUR_DATASET}

mkdir -p ${OUTPUT_DIR}

past_num=$((TASK_ID - 1))
config_files_str=""
checkpoint_base_dir="./checkpoints/Qwen3-VL-2B/RegLoRA-CL/training"
for ((i=0; i<past_num; i++)); do
    dataset_name=${DATASETS[$i]}
    ckpt_path="${checkpoint_base_dir}/${dataset_name}"
    if [ -z "$config_files_str" ]; then
        config_files_str="$ckpt_path"
    else
        config_files_str="$config_files_str,$ckpt_path"
    fi
done
reg_lamda=25
mask_ratio=0.02
echo "======================================"
echo "config_files: $config_files_str"
echo "reg_lamda: $reg_lamda | mask_ratio: $mask_ratio"
echo "======================================"

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
# If you want to set the min pixels and max pixels for Qwen3-VL, You should set as (N * 32 * 32)

deepspeed src/train/train_sft.py \
    --use_liger_kernel False \
    --lora_enable True \
    --vision_lora True \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 128 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --image_folder $MEDIA_DIR \
    --prompt_path src/dataset/prompts_1.yaml \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((128 * 32 * 32)) \
    --image_max_pixels $((512 * 32 * 32)) \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing False \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4 \
    --use_reglora True \
    --config_files "$config_files_str" \
    --reg_lamda $reg_lamda \
    --mask_ratio $mask_ratio \
    --lora_target "down_proj,q_proj,v_proj,o_proj,gate_proj,up_proj,k_proj,attn.qkv,attn.proj" 

echo "======================================"
echo "Merging LoRA to Weights for: $CUR_DATASET"
echo "======================================"
python src/merge_lora_weights.py \
    --model-path $OUTPUT_DIR \
    --model-base $MODEL_NAME  \
    --save-model-path $MERGE_SAVE_PATH \
    --safe-serialization
