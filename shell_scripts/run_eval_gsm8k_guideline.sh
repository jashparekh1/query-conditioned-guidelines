#!/bin/bash

set -euo pipefail

# 1) Launch servers (2 GPUs each) in background
# Customize these paths/ports/names as needed.

# Guide LLM
GUIDE_MODEL_PATH=${GUIDE_MODEL_PATH:-"merge_models/546-nlp-gsm8k-grpo-qwen-2.5-1.5b-instruct-guidelines"}
GUIDE_PORT=${GUIDE_PORT:-1226}
GUIDE_NAME=${GUIDE_NAME:-guide-llm}
GUIDE_GPUS=${GUIDE_GPUS:-"4,5"}

# Base LLM
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"/shared/data2/jiashuo5/pretrain_models/Qwen2.5-1.5B-Instruct"}
BASE_PORT=${BASE_PORT:-1225}
BASE_NAME=${BASE_NAME:-base-llm}
BASE_GPUS=${BASE_GPUS:-"6,7"}

# Dataset dir (produced by examples/data_preprocess/guidelines.py or gsm8k.py)
DATASET_DIR=${DATASET_DIR:-"data/gsm8k"}

# Output file
OUTPUT_PATH=${OUTPUT_PATH:-"gsm8k_guideline_eval_results.jsonl"}

# echo "Starting guide LLM server on GPUs ${GUIDE_GPUS} port ${GUIDE_PORT}..."
# CUDA_VISIBLE_DEVICES=${GUIDE_GPUS} PORT=${GUIDE_PORT} MODEL_PATH=${GUIDE_MODEL_PATH} MODEL_NAME=${GUIDE_NAME} \
#     nohup bash /shared/data2/jiashuo5/verl/serve_guide_llm.sh > guide_server.log 2>&1 &
# GUIDE_PID=$!

# echo "Starting base LLM server on GPUs ${BASE_GPUS} port ${BASE_PORT}..."
# CUDA_VISIBLE_DEVICES=${BASE_GPUS} PORT=${BASE_PORT} MODEL_PATH=${BASE_MODEL_PATH} MODEL_NAME=${BASE_NAME} \
#     nohup bash /shared/data2/jiashuo5/verl/serve_base_llm.sh > base_server.log 2>&1 &
# BASE_PID=$!

# echo "Waiting 15s for servers to become ready..."
# sleep 15

echo "Running evaluation..."
PYTHONUNBUFFERED=1 python3 -m eval.gsm8k_guideline_eval \
    --dataset_path "${DATASET_DIR}" \
    --split test \
    --guide_base_url "http://127.0.0.1:${GUIDE_PORT}/v1" \
    --guide_model_name "${GUIDE_NAME}" \
    --solver_base_url "http://127.0.0.1:${BASE_PORT}/v1" \
    --solver_model_name "${BASE_NAME}" \
    --output_path "${OUTPUT_PATH}"

echo "Eval finished. Stopping servers..."
kill ${GUIDE_PID} || true
kill ${BASE_PID} || true

echo "Done. Results at ${OUTPUT_PATH}"


