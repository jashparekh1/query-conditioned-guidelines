#!/bin/bash
# Path configuration for GRPO training

# Project root
export PROJECT_ROOT=/projects/bfgx/jparekh/query-conditioned-guidelines

# External dependencies
export VERL_PATH=/projects/bfgx/jparekh/test_ngc/verl
export VLLM_PATH=/projects/bfgx/jparekh/test_ngc/vllm

# Add to Python path
export PYTHONPATH=$VERL_PATH:$VLLM_PATH:$PYTHONPATH

# Data paths  
export GSM8K_TRAIN=$PROJECT_ROOT/data/gsm8k/train.parquet
export GSM8K_TEST=$PROJECT_ROOT/data/gsm8k/test.parquet

# Model
export QWEN_1_5B=Qwen/Qwen2.5-1.5B-Instruct
export QWEN_3B=Qwen/Qwen2.5-3B-Instruct
export QWEN_7B=Qwen/Qwen2.5-7B-Instruct

# Cache directory
export HF_HOME=/projects/bfgx/jparekh/causal-inference-project/v2/cache

# WandB
export WANDB_API_KEY=705ca850b67f54b4b59da03f625d5751d9b984d1

echo "================================"
echo "Project: $PROJECT_ROOT"
echo "VERL: $VERL_PATH"
echo "vLLM: $VLLM_PATH"
echo "Train: $GSM8K_TRAIN"
echo "Model 1.5B: $QWEN_1_5B"
echo "Model 3B: $QWEN_3B"
echo "================================"
