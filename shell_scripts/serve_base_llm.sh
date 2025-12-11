#!/bin/bash

# Base LLM OpenAI-compatible server (vLLM) on 2 GPUs
# Configurable via env vars:
#   MODEL_PATH: HF or local path to the base model (e.g., Qwen2.5-3B-Instruct)
#   CUDA_VISIBLE_DEVICES: GPU ids, default "2,3"
#   PORT: server port, default 1225
#   MODEL_NAME: served model name, default "base-llm"

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"/shared/data3/xzhong23/models/Qwen2.5-1.5B-Instruct"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
PORT=${PORT:-1225}
MODEL_NAME=${MODEL_NAME:-base-llm}

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size 2 \
    --port ${PORT} \
    --disable-custom-all-reduce \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192


