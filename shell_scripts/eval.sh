#!/bin/bash
set -euo pipefail

# Use singularity container (same as training scripts) to avoid GLIBC compatibility issues
CONTAINER=/projects/bfgx/jparekh/test_ngc/torch2501.sif
PROJECT_ROOT=/projects/bfgx/jparekh/query-conditioned-guidelines

# Set HuggingFace cache to /tmp/jparekh
mkdir -p /tmp/jparekh
export HF_HOME=/tmp/jparekh
export TRANSFORMERS_CACHE=/tmp/jparekh
export HF_DATASETS_CACHE=/tmp/jparekh

# Use 1 GPU (sufficient for 1.5B model) or 4 GPUs (faster)
# For 1 GPU: CUDA_VISIBLE_DEVICES=0 and tensor_parallel_size=1
# For 4 GPUs: CUDA_VISIBLE_DEVICES=0,1,2,3 and tensor_parallel_size=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}

echo "Using HuggingFace cache: /tmp/jparekh"
echo "Using CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Using tensor_parallel_size: $TENSOR_PARALLEL_SIZE"

cd /projects/bfgx/jparekh/test_ngc

singularity exec --nv \
  --bind /projects/bfgx/jparekh:/projects/bfgx/jparekh \
  --bind /dev/shm \
  --env HF_HOME=/tmp/jparekh \
  --env TRANSFORMERS_CACHE=/tmp/jparekh \
  --env HF_DATASETS_CACHE=/tmp/jparekh \
  --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  --env PYTHONUNBUFFERED=1 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  $CONTAINER bash -c "
    source verl_env/bin/activate
    cd $PROJECT_ROOT
    python eval.py \
      --guilder_model merge_models/offline-vllm-1.5b-step-498 \
      --solver_model Qwen/Qwen2.5-1.5B-Instruct \
      --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
      --split test \
      --out_dir outputs/gsm8k_two_stage_outputs
  "
