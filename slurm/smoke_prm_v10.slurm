#!/bin/bash
#SBATCH --account=bfgx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=smoke-prm-v10
#SBATCH --chdir=/projects/bfgx/jparekh/query-conditioned-guidelines
#SBATCH --output=experiments/logs/smoke_prm_v10_%j.out
#SBATCH --error=experiments/logs/smoke_prm_v10_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --reservation=update

# Smoke test: 2 steps of PRM training to verify NCCL works.

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
  echo "ERROR: SIF not found."
  exit 1
fi

mkdir -p "$WORK_NVME/hf_cache" "$WORK_NVME/apptainer_cache"

echo "=========================================="
echo "SMOKE TEST: PRM v10 (2 steps)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

CKPT_NAME="smoke-prm-v10"
rm -rf "$WORK_NVME/checkpoints/$CKPT_NAME"

unset HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE 2>/dev/null || true

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
  --env NCCL_DEBUG=INFO \
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
  --env VERL_DIST_TIMEOUT_SECONDS="${VERL_DIST_TIMEOUT_SECONDS:-600}" \
  --env VERL_CHECKPOINT_DIR="/nvme/checkpoints/$CKPT_NAME" \
  --env VERL_SAVE_FREQ=100 \
  --env VERL_GUIDELINES_LOG_DIR="${VERL_GUIDELINES_LOG_DIR:-/workspace/query-conditioned-guidelines/experiments/logs/solver_debug_smoke}" \
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

    mkdir -p /nvme/checkpoints/'"$CKPT_NAME"'
    export VERL_EXPERIMENT_SUFFIX="-smoke-prm-v10"
    export VERL_TOTAL_STEPS=2
    export VERL_TEST_FREQ=100
    export VERL_ROLLOUT_N=16
    export MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
    export DATA_DIR="/workspace/query-conditioned-guidelines/experiments/data/dapomath_3k_prm"
    export VERL_GUIDELINES_BASELINES_PATH="/workspace/query-conditioned-guidelines/experiments/data/dapomath_30k/baselines.json"
    export VERL_GUIDELINES_PRM_BASELINES_PATH="/workspace/query-conditioned-guidelines/experiments/data/dapomath_3k_prm/prm_baselines.json"
    export VERL_VAL_FILE="/workspace/query-conditioned-guidelines/experiments/data/dapomath_3k_prm/test.parquet"
    bash experiments/run_train.sh \
      trainer.experiment_name=smoke-prm-v10 \
      trainer.logger=console \
      +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NET=Socket \
      +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NET_PLUGIN="" \
      +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="" \
      +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME="" \
      +ray_kwargs.ray_init.runtime_env.env_vars.FI_PROVIDER="" \
      +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG=INFO
  '

echo "=========================================="
echo "Smoke test finished."
echo "=========================================="
