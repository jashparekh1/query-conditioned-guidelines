#!/bin/bash

# Guide LLM OpenAI-compatible server (vLLM) on 2 GPUs
# Configurable via env vars:
#   MODEL_PATH: HF or local path to the trained guide model
#   CUDA_VISIBLE_DEVICES: GPU ids, default "0,1"
#   PORT: server port, default 1226
#   MODEL_NAME: served model name, default "guide-llm"

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"/path/to/your/trained-guide-model"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
PORT=${PORT:-1226}
MODEL_NAME=${MODEL_NAME:-guide-llm}

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size 2 \
    --port ${PORT} \
    --disable-custom-all-reduce \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192


