#!/bin/bash
set -euo pipefail
MAIN_LOG=./zeroshot.log

exec > >(tee -a "$MAIN_LOG") 2>&1
BASE_MODEL=/your_model_path/Qwen/Qwen3-VL-2B-Instruct
BASE_PATH=/your_data_path

DATASETS=(
    "FinMME"
    "We-Math2"
    "Navigation"
    "MedBookVQA"
    "Puzzle"
)

BASE_RESULTS_DIR=./results/Qwen3-VL-2B/ZeroShot

DISABLE_FLASH_ATTN2=true
BATCH_SIZE=2048
echo "======================================"
echo "Starting Zeroshot Evaluation"
echo "======================================"
for idx in "${!DATASETS[@]}"; do
    TEST_DATASET=${DATASETS[$idx]}   
    TEST_FILE=${BASE_PATH}/${TEST_DATASET}/jsons/test/data.json
    MEDIA_DIR=${BASE_PATH}/${TEST_DATASET}/images
    RESULTS_DIR="${BASE_RESULTS_DIR}/${TEST_DATASET}"
    mkdir -p "${RESULTS_DIR}"

    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -s -d,)
        export CUDA_VISIBLE_DEVICES=$ALL_GPUS
    fi

    IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
    CHUNKS=${#GPULIST[@]}

    echo "======================================"
    echo "Starting evaluation"
    echo "======================================"
    echo "Base model: ${BASE_MODEL}"
    echo "Test file: ${TEST_FILE}"
    echo "Image directory: ${MEDIA_DIR}"
    echo "Results directory: ${RESULTS_DIR}"
    echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
    echo "GPU count: ${CHUNKS}"
    echo "======================================"

    mkdir -p "${RESULTS_DIR}"
    echo "======================================"
    
    FLASH_ATTN_FLAG=""
    if [ "${DISABLE_FLASH_ATTN2}" = "true" ]; then
        FLASH_ATTN_FLAG="--disable_flash_attn2"
    fi

    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    python src/eval/inference.py \
        --base_model "${BASE_MODEL}" \
        --test_file "${TEST_FILE}" \
        --media_dir "${MEDIA_DIR}" \
        --output_dir "${RESULTS_DIR}" \
        --prompts_file "src/dataset/prompts_2.yaml" \
        --max_completion_length 2048 \
        --tensor_parallel_size ${CHUNKS} \
        --batch_size ${BATCH_SIZE} \
        ${FLASH_ATTN_FLAG}

    OUTPUT_FILE="${RESULTS_DIR}/merge.jsonl"

    python src/eval/eval.py \
        --dataset_name "${TEST_DATASET}" \
        --merged_file "${OUTPUT_FILE}" \
        --output_dir "${RESULTS_DIR}"

    if [ $? -eq 0 ]; then
        echo "======================================"
        echo "Evaluation completed."
        echo "======================================"
    else
        echo "======================================"
        echo "Evaluation failed."
        echo "======================================"
        exit 1
    fi

done
