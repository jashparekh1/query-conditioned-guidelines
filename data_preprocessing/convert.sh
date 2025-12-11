#!/bin/bash
set -euo pipefail

# Load modules to get torch
echo "Loading modules..."
module load cuda
module load python/3.11.9
module load python/miniforge3_pytorch/2.7.0

# Add local verl to PYTHONPATH (merge script imports directly from verl/utils/tokenizer.py)
export PYTHONPATH=/projects/bfgx/jparekh/query-conditioned-guidelines:${PYTHONPATH:-}

# Verify torch is available
if python -c "import torch" 2>/dev/null; then
    echo "✓ torch is available"
    python -c "import torch; print(f'  torch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"
else
    echo "✗ ERROR: torch is not available"
    exit 1
fi

# Set HuggingFace cache to /tmp/jparekh
mkdir -p /tmp/jparekh
export HF_HOME=/tmp/jparekh
export TRANSFORMERS_CACHE=/tmp/jparekh
export HF_DATASETS_CACHE=/tmp/jparekh
echo "Using HuggingFace cache: /tmp/jparekh"

TRAIN_MODEL_NAME=checkpoints/546-nlp-gsm8k-grpo-offline-vllm/offline-vllm-1.5b-1118-2134-resume/global_step_498/actor
HF_CONFIG_PATH="$TRAIN_MODEL_NAME/huggingface"
TARGET_DIR=merge_models/offline-vllm-1.5b-step-498

python scripts/merge_model.py merge --backend fsdp \
    --local_dir "$TRAIN_MODEL_NAME" \
    --hf_model_path "$HF_CONFIG_PATH" \
    --target_dir "$TARGET_DIR"