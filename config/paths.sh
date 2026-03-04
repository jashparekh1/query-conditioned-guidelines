#!/bin/bash
# Path configuration for GRPO training.
# Only used outside the container (local dev). Inside the container,
# run_train.sh detects /opt/verl and ignores this file.

export PROJECT_ROOT=/projects/bfgx/jparekh/query-conditioned-guidelines

# External dependencies (only needed if not using the container)
export VERL_PATH=${VERL_PATH:-/opt/verl}
export VLLM_PATH=${VLLM_PATH:-}

export PYTHONPATH=$PROJECT_ROOT:$VERL_PATH:${VLLM_PATH:+$VLLM_PATH:}$PYTHONPATH

# Cache directory
export HF_HOME=${HF_HOME:-/tmp/hf_cache}

# WandB: set WANDB_API_KEY in your environment, not here.
