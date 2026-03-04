#!/bin/bash
#SBATCH --account=bfgx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=train-prm-v10
#SBATCH --chdir=/projects/bfgx/jparekh/query-conditioned-guidelines
#SBATCH --output=experiments/logs/train_prm_v10_%j.out
#SBATCH --error=experiments/logs/train_prm_v10_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --reservation=update

# v10: PRM-based reward + 4B planner on DAPO-Math 30k.
# Planner: Qwen3-4B-Instruct-2507 (trainable via GRPO)
# Solver:  Qwen2.5-14B-Instruct-AWQ (frozen, on GPU 3)
# PRM:     Qwen2.5-Math-PRM-7B (frozen, on GPU 3, loaded after solver)
# Reward:  R = min(PRM_with_plan) - min(PRM_without_plan) - boxed_penalty
# Data:    DAPO-Math 30k (experiments/data/dapomath_30k/)
# 50-step trial to validate PRM reward signal before longer run.

set -e
PROJECT_ROOT="/projects/bfgx/jparekh/query-conditioned-guidelines"
cd "$PROJECT_ROOT"
mkdir -p experiments/logs

USER="${USER:-jparekh}"
WORK_NVME="/work/nvme/bfgx/$USER"
SIF_REPO="$PROJECT_ROOT/containers/verl-vllm-24.04.sif"
if [[ -f "$SIF_REPO" ]]; then
  SIF="$SIF_REPO"
else
  SIF="$WORK_NVME/verl-vllm-24.04.sif"
fi

if [[ ! -f "$SIF" ]]; then
  echo "ERROR: SIF not found. Tried: $SIF_REPO and $WORK_NVME/verl-vllm-24.04.sif"
  exit 1
fi

if ! mkdir -p "$WORK_NVME/hf_cache" "$WORK_NVME/apptainer_cache"; then
  echo "ERROR: Cannot create $WORK_NVME (or subdirs)."
  exit 1
fi
if ! touch "$WORK_NVME/hf_cache/.write_test" 2>/dev/null; then
  echo "ERROR: $WORK_NVME is not writable."
  exit 1
fi
rm -f "$WORK_NVME/hf_cache/.write_test"

echo "=========================================="
echo "v10: PRM reward + 4B planner on DAPO-Math"
echo "batch=60, rollout_n=16, lr=1e-6"
echo "Planner: Qwen3-4B-Instruct-2507"
echo "Solver:  Qwen2.5-14B-Instruct-AWQ"
echo "PRM:     Qwen2.5-Math-PRM-7B"
echo "Reward:  R = min(PRM) - boxed_penalty"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "SIF: $SIF"
echo "=========================================="

CKPT_NAME="dapomath30k-4b-50steps-prm-v10"
if [[ "${VERL_CLEAN_START:-1}" =~ ^(1|true|yes)$ ]]; then
  echo "VERL_CLEAN_START=1: Removing NVMe checkpoint to start fresh..."
  rm -rf "$WORK_NVME/checkpoints/$CKPT_NAME"
fi

# Resume: if project has a checkpoint and NVMe doesn't, copy project -> NVMe
PROJECT_CKPT="$PROJECT_ROOT/checkpoints/query-conditioned-guidelines/$CKPT_NAME"
NVME_CKPT="$WORK_NVME/checkpoints/$CKPT_NAME"
if [[ -f "$PROJECT_CKPT/latest_checkpointed_iteration.txt" ]]; then
  STEP=$(cat "$PROJECT_CKPT/latest_checkpointed_iteration.txt" | tr -d '[:space:]')
  if [[ -n "$STEP" && -d "$PROJECT_CKPT/global_step_$STEP" ]]; then
    if [[ ! -f "$NVME_CKPT/latest_checkpointed_iteration.txt" ]] || [[ "$(cat "$NVME_CKPT/latest_checkpointed_iteration.txt" | tr -d '[:space:]')" -lt "$STEP" ]]; then
      echo "Resuming: copying checkpoint step $STEP from project to NVMe..."
      mkdir -p "$NVME_CKPT"
      cp -r "$PROJECT_CKPT/global_step_$STEP" "$NVME_CKPT/"
      cp "$PROJECT_CKPT/latest_checkpointed_iteration.txt" "$NVME_CKPT/"
      echo "Done. Will resume from step $STEP."
    fi
  fi
fi

unset HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE 2>/dev/null || true

TRAIN_EXIT=0
# Hide container's CUDA compat lib so the host's (newer) driver is used
EMPTY_DIR="$WORK_NVME/empty_cuda_compat"
mkdir -p "$EMPTY_DIR"

apptainer exec --nv \
  --bind "/projects/bfgx/jparekh:/workspace" \
  --bind "$WORK_NVME:/nvme" \
  --bind "$WORK_NVME/apptainer_cache:/root/.cache" \
  --bind "$EMPTY_DIR:/usr/local/cuda/compat/lib" \
  --env NCCL_NET=Socket \
  --env NCCL_NET_PLUGIN="" \
  --env NCCL_PLUGIN_P2P="" \
  --env NCCL_IB_DISABLE="" \
  --env NCCL_SOCKET_IFNAME="" \
  --env NCCL_CROSS_NIC="" \
  --env FI_PROVIDER="" \
  --env VERL_PATH=/opt/verl \
  --env VLLM_TORCH_DTYPE=bfloat16 \
  --env TORCH_DTYPE=bfloat16 \
  --env HF_HOME=/tmp/hf_cache \
  --env HF_HUB_CACHE=/tmp/hf_cache \
  --env TRANSFORMERS_CACHE=/tmp/hf_cache \
  --env HF_DATASETS_CACHE=/tmp/hf_cache \
  --env XDG_CACHE_HOME=/tmp/xdg_cache \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real \
  --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env TMPDIR=/tmp \
  --env RAY_TMPDIR=/tmp \
  --env VERL_GUIDELINES_USE_MATH_VERIFY="${VERL_GUIDELINES_USE_MATH_VERIFY:-1}" \
  --env VERL_DIST_TIMEOUT_SECONDS="${VERL_DIST_TIMEOUT_SECONDS:-1800}" \
  --env VERL_CHECKPOINT_DIR="/nvme/checkpoints/$CKPT_NAME" \
  --env VERL_SAVE_FREQ=50 \
  --env VERL_GUIDELINES_LOG_DIR="${VERL_GUIDELINES_LOG_DIR:-/workspace/query-conditioned-guidelines/experiments/logs/solver_debug_v10}" \
  --env VERL_GUIDELINES_USE_PRM=1 \
  --env VERL_GUIDELINES_PRM_MODEL="Qwen/Qwen2.5-Math-PRM-7B" \
  --env VERL_GUIDELINES_PRM_LAMBDA=0.5 \
  --env VERL_GUIDELINES_PRM_SOFTMIN_TEMP=0.1 \
  --env VERL_GUIDELINES_BOXED_PENALTY=0.5 \
  --env VERL_GUIDELINES_PRM_BASELINES_PATH="/workspace/query-conditioned-guidelines/experiments/data/dapomath_3k_prm/prm_baselines.json" \
  --env VERL_GUIDELINES_SOLVER_GPU_MEM=0.35 \
  --env VLLM_USE_V1=0 \
  "$SIF" \
  bash -c '
    source /workspace/setup_verl_apptainer.sh 2>/dev/null || true
    cd /workspace/query-conditioned-guidelines
    pip install -q math-verify || { echo "ERROR: pip install math-verify failed"; exit 1; }
    export PYTHONPATH="/workspace/query-conditioned-guidelines:${PYTHONPATH}"

    # Step 0: Create 3k training subset matching PRM baselines
    SUBSET_DIR="experiments/data/dapomath_3k_prm"
    echo "=== Creating 3k training subset from PRM baselines ==="
    python -m experiments.create_prm_subset \
      --baselines experiments/data/dapomath_30k/prm_baselines.json \
      --input experiments/data/dapomath_30k/train.parquet \
      --output "$SUBSET_DIR/train.parquet"

    # Copy test.parquet for validation
    cp experiments/data/dapomath_30k/test.parquet "$SUBSET_DIR/test.parquet"

    # Step 1: Train
    mkdir -p /nvme/checkpoints/'"$CKPT_NAME"'
    export VERL_EXPERIMENT_SUFFIX="-prm-v10"
    export VERL_TOTAL_STEPS=100
    export VERL_TEST_FREQ=10
    export VERL_ROLLOUT_N=16
    export MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
    export DATA_DIR="/workspace/query-conditioned-guidelines/$SUBSET_DIR"
    export VERL_GUIDELINES_BASELINES_PATH="/workspace/query-conditioned-guidelines/experiments/data/dapomath_30k/baselines.json"
    export VERL_GUIDELINES_PRM_BASELINES_PATH="/workspace/query-conditioned-guidelines/$SUBSET_DIR/prm_baselines.json"
    export VERL_VAL_FILE="/workspace/query-conditioned-guidelines/$SUBSET_DIR/test.parquet"
    bash experiments/run_train.sh \
      trainer.experiment_name=dapomath3k-4b-50steps-prm-v10 \
      +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NET=Socket \
      +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NET_PLUGIN="" \
      +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="" \
      +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME="" \
      +ray_kwargs.ray_init.runtime_env.env_vars.FI_PROVIDER=""
  ' || TRAIN_EXIT=$?

# Copy last checkpoint from NVMe to project when job ends
NVME_CKPT="$WORK_NVME/checkpoints/$CKPT_NAME"
PROJECT_CKPT="$PROJECT_ROOT/checkpoints/query-conditioned-guidelines/$CKPT_NAME"
if [[ -f "$NVME_CKPT/latest_checkpointed_iteration.txt" ]]; then
  STEP=$(cat "$NVME_CKPT/latest_checkpointed_iteration.txt" | tr -d '[:space:]')
  if [[ -n "$STEP" && -d "$NVME_CKPT/global_step_$STEP" ]]; then
    echo "Copying last checkpoint (step $STEP) from NVMe to project..."
    mkdir -p "$PROJECT_CKPT"
    cp -r "$NVME_CKPT/global_step_$STEP" "$PROJECT_CKPT/"
    cp "$NVME_CKPT/latest_checkpointed_iteration.txt" "$PROJECT_CKPT/"
    echo "Done. Resumable from $PROJECT_CKPT"
  fi
fi

echo "=========================================="
echo "Job finished."
echo "=========================================="
exit $TRAIN_EXIT
