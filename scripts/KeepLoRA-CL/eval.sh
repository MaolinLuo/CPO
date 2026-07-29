#!/bin/bash
set -euo pipefail

BASE_PATH=$1
TASK_ID=$2
shift 2
DATASETS=("$@")

CUR_DATASET=${DATASETS[$((TASK_ID - 1))]}

BASE_MODEL_DIR=./checkpoints/Qwen3-VL-2B/KeepLoRA-CL/model
BASE_RESULTS_DIR=./results/Qwen3-VL-2B/KeepLoRA-CL

BATCH_SIZE=256
DISABLE_FLASH_ATTN2=false
TEST_DATA_DIR="test"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -s -d,)
    export CUDA_VISIBLE_DEVICES=$ALL_GPUS
fi

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}

MODEL_NAME="${BASE_MODEL_DIR}/${CUR_DATASET}"

echo "======================================"
echo "Evaluating after Task ${TASK_ID}: ${CUR_DATASET}"
echo "Model: ${MODEL_NAME}"
echo "======================================"

for TEST_TASK_ID in $(seq 1 ${TASK_ID})
do
    TEST_DATASET=${DATASETS[$((TEST_TASK_ID - 1))]}
    TEST_FILE=${BASE_PATH}/${TEST_DATASET}/jsons/test/data.json
    MEDIA_DIR=${BASE_PATH}/${TEST_DATASET}/images
    RESULTS_DIR="${BASE_RESULTS_DIR}/${CUR_DATASET}/${TEST_DATASET}"
    
    mkdir -p "${RESULTS_DIR}"

    FLASH_ATTN_FLAG=""
    if [ "${DISABLE_FLASH_ATTN2}" = "true" ]; then
        FLASH_ATTN_FLAG="--disable_flash_attn2"
    fi

    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    python src/eval/inference.py \
        --base_model "${MODEL_NAME}" \
        --test_file "${TEST_FILE}" \
        --media_dir "${MEDIA_DIR}" \
        --output_dir "${RESULTS_DIR}" \
        --prompts_file "src/dataset/prompts_1.yaml" \
        --max_completion_length 2048 \
        --tensor_parallel_size ${CHUNKS} \
        --batch_size ${BATCH_SIZE} \
        ${FLASH_ATTN_FLAG}

    OUTPUT_FILE="${RESULTS_DIR}/merge.jsonl"

    python src/eval/eval.py \
        --dataset_name "${TEST_DATASET}" \
        --merged_file "${OUTPUT_FILE}" \
        --output_dir "${RESULTS_DIR}"

done

echo "======================================"
echo "Evaluation completed for Task ${TASK_ID}"
echo "======================================"
