#!/bin/bash
set -euo pipefail
LOG_FILE="./results/Qwen3-VL-2B/Zeroshot_OOD/eval.log"
mkdir -p ./results/Qwen3-VL-2B/Zeroshot_OOD
exec > >(tee -a "$LOG_FILE") 2>&1
DATA=(
    "MMMU_pro"
    "MathVerse"
    "RealworldQA"
    "MMStar"
    "CountBenchQA"
    "DocVQA"
    "Charxiv"
    "POPE"
    "MathVision"
    "MathVista"
)

MODEL_NAME=/your_model_path/Qwen/Qwen3-VL-2B-Instruct
TEST_FILE_BASE=/your_data_path/Eval_OOD_Datasets
RESULTS_DIR_BASE=./results/Qwen3-VL-2B/Zeroshot_OOD

BATCH_SIZE=2048
DISABLE_FLASH_ATTN2=true
for dataset in "${DATA[@]}"; do
    if [ "${dataset}" == "POPE" ]; then
        TEST_FILE="${TEST_FILE_BASE}/POPE/coco_pope.json"
        MEDIA_DIR="${TEST_FILE_BASE}/POPE/coco/image"
    elif [[ "${dataset}" =~ ^(We-Math2|Chemistry|Coding|Navigation|CVQA|FinMME|MedBookVQA|InstructFollow)$ ]]; then
        TEST_FILE="${TEST_FILE_BASE}/${dataset}/jsons/test/data.json"
        MEDIA_DIR="${TEST_FILE_BASE}/${dataset}/images"
    else
        # parquet-based HuggingFace datasets (MMMU_pro, MathVerse, RealworldQA, MMStar,
        # CountBenchQA, DocVQA, Charxiv, MathVision, MathVista, …)
        TEST_FILE="${TEST_FILE_BASE}/${dataset}"
        MEDIA_DIR=""
    fi
    RESULTS_DIR="${RESULTS_DIR_BASE}/${dataset}"
    if [ "${dataset}" == "MMMU_pro" ]; then
        Options=(
                "standard (4 options)"
                "standard (10 options)"
                "vision"
                )
        for options in "${Options[@]}"; do
            TEST_FILE_OPTION="${TEST_FILE}/${options}"
            RESULTS_DIR_OPTION="${RESULTS_DIR}/${options}"
            echo "========================================"
            echo "Processing dataset: ${dataset}"
            echo "========================================"
            if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
                    ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -s -d,)
                    export CUDA_VISIBLE_DEVICES=$ALL_GPUS
                fi

            IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
            CHUNKS=${#GPULIST[@]}

            echo "======================================"
            echo "Starting evaluation"
            echo "======================================"
            echo "Model: ${MODEL_NAME}"
            echo "Test file: ${TEST_FILE_OPTION}"
            echo "Results directory: ${RESULTS_DIR_OPTION}"
            echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
            echo "GPU count: ${CHUNKS}"
            echo "======================================"

            mkdir -p "${RESULTS_DIR_OPTION}"
            echo "======================================"
            FLASH_ATTN_FLAG=""
            if [ "${DISABLE_FLASH_ATTN2}" = "true" ]; then
                FLASH_ATTN_FLAG="--disable_flash_attn2"
            fi

            export VLLM_WORKER_MULTIPROC_METHOD=spawn
            python src/eval/inference.py \
                --base_model "${MODEL_NAME}" \
                --test_file "${TEST_FILE_OPTION}" \
                --media_dir "${MEDIA_DIR}" \
                --output_dir "${RESULTS_DIR_OPTION}" \
                --prompts_file "src/dataset/prompts_ood.yaml" \
                --max_completion_length 16384 \
                --tensor_parallel_size ${CHUNKS} \
                --batch_size ${BATCH_SIZE} \
                ${FLASH_ATTN_FLAG}

            OUTPUT_FILE="${RESULTS_DIR_OPTION}/merge.jsonl"

            echo "Merging completed!"
            echo "Evaluation completed. Results saved to ${OUTPUT_FILE}"
            echo "======================================"

            python src/eval/eval.py \
                --dataset_name "${dataset}" \
                --merged_file "${OUTPUT_FILE}" \
                --output_dir "${RESULTS_DIR_OPTION}"
        done
    else
        echo "========================================"
        echo "Processing dataset: ${dataset}"
        echo "========================================"
        if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
                ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -s -d,)
                export CUDA_VISIBLE_DEVICES=$ALL_GPUS
            fi

        IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
        CHUNKS=${#GPULIST[@]}

        echo "======================================"
        echo "Starting evaluation"
        echo "======================================"
        echo "Model: ${MODEL_NAME}"
        echo "Test file: ${TEST_FILE}"
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
            --base_model "${MODEL_NAME}" \
            --test_file "${TEST_FILE}" \
            --media_dir "${MEDIA_DIR}" \
            --output_dir "${RESULTS_DIR}" \
            --prompts_file "src/dataset/prompts_ood.yaml" \
            --max_completion_length 16384 \
            --tensor_parallel_size ${CHUNKS} \
            --batch_size ${BATCH_SIZE} \
            ${FLASH_ATTN_FLAG}

        OUTPUT_FILE="${RESULTS_DIR}/merge.jsonl"

        echo "Merging completed!"
        echo "Evaluation completed. Results saved to ${OUTPUT_FILE}"
        echo "======================================"
        
        python src/eval/eval.py \
            --dataset_name "${dataset}" \
            --merged_file "${OUTPUT_FILE}" \
            --output_dir "${RESULTS_DIR}"
        
    fi
done
