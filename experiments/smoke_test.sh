#!/usr/bin/env bash
# Quick smoke test: 2 training steps on the active compute node.
# Assumes you already have a GPU allocation (srun --pty bash or inside sbatch).
#
# Usage:
#   srun --account=bfgx-dtai-gh --partition=ghx4 --reservation=update \
#        --gres=gpu:nvidia_gh200_120gb:4 --time=00:20:00 --mem=200G --cpus-per-task=32 --pty bash
#   bash experiments/smoke_test.sh
#
# Or as a one-liner from login node:
#   srun --account=bfgx-dtai-gh --partition=ghx4 --reservation=update \
#        --gres=gpu:nvidia_gh200_120gb:4 --time=00:20:00 --mem=200G --cpus-per-task=32 \
#        bash experiments/smoke_test.sh

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Check we have GPUs
if ! command -v nvidia-smi &>/dev/null && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "ERROR: No GPUs detected. Run this on a compute node with GPUs allocated."
  echo "  srun --account=bfgx-dtai-gh --partition=ghx4 --reservation=update \\"
  echo "       --gres=gpu:nvidia_gh200_120gb:4 --time=00:20:00 --mem=200G --cpus-per-task=32 --pty bash"
  exit 1
fi

# Find container
SIF="$PROJECT_ROOT/containers/verl-vllm-24.04.sif"
if [ ! -f "$SIF" ]; then
  SIF="/work/nvme/bfgx/${USER}/verl-vllm-24.04.sif"
fi
if [ ! -f "$SIF" ]; then
  echo "ERROR: Container not found. Tried:"
  echo "  $PROJECT_ROOT/containers/verl-vllm-24.04.sif"
  echo "  /work/nvme/bfgx/${USER}/verl-vllm-24.04.sif"
  exit 1
fi

# Prepare dirs
WORK_NVME="/work/nvme/bfgx/${USER}"
mkdir -p "$WORK_NVME/apptainer_cache" "$WORK_NVME/empty_cuda_compat" /tmp/smoke_ckpt

echo "=========================================="
echo "Smoke test: 2 training steps"
echo "Node: $(hostname)"
echo "Container: $SIF"
echo "=========================================="

apptainer exec --nv \
  --bind /projects/bfgx/jparekh:/workspace \
  --bind "$WORK_NVME:/nvme" \
  --bind "$WORK_NVME/apptainer_cache:/root/.cache" \
  --bind "$WORK_NVME/empty_cuda_compat:/usr/local/cuda/compat/lib" \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONUNBUFFERED=1 \
  --env NCCL_NET=Socket \
  --env NCCL_NET_PLUGIN="" \
  --env NCCL_PLUGIN_P2P="" \
  --env NCCL_IB_DISABLE="" \
  --env NCCL_SOCKET_IFNAME="" \
  --env NCCL_CROSS_NIC="" \
  --env FI_PROVIDER="" \
  --env VLLM_USE_V1=0 \
  --env VERL_PATH=/opt/verl \
  --env VLLM_TORCH_DTYPE=bfloat16 \
  --env HF_HOME=/tmp/hf_cache \
  --env HF_HUB_CACHE=/tmp/hf_cache \
  --env TRANSFORMERS_CACHE=/tmp/hf_cache \
  --env HF_DATASETS_CACHE=/tmp/hf_cache \
  --env XDG_CACHE_HOME=/tmp/xdg_cache \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  --env TMPDIR=/tmp \
  --env RAY_TMPDIR=/tmp \
  --env VERL_GUIDELINES_USE_MATH_VERIFY=1 \
  --env VERL_GUIDELINES_SOLVER_GPU_MEM=0.35 \
  --env VERL_DIST_TIMEOUT_SECONDS=600 \
  "$SIF" \
  bash -c '
    source /workspace/setup_verl_apptainer.sh 2>/dev/null || true
    cd /workspace/query-conditioned-guidelines
    pip install -q math-verify || { echo "ERROR: pip install math-verify failed"; exit 1; }
    export PYTHONPATH="/workspace/query-conditioned-guidelines:${PYTHONPATH}"

    VERL_TOTAL_STEPS=2 \
    VERL_TEST_FREQ=100 \
    VERL_SAVE_FREQ=100 \
    VERL_ROLLOUT_N=16 \
    VERL_EXPERIMENT_SUFFIX=-smoke \
    VERL_CHECKPOINT_DIR=/tmp/smoke_ckpt \
    MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 \
    DATA_DIR=/workspace/query-conditioned-guidelines/experiments/data/dapomath_3k \
    bash experiments/run_train.sh \
      trainer.experiment_name=smoke \
      trainer.logger=console \
      +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NET=Socket \
      +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NET_PLUGIN="" \
      +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="" \
      +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME="" \
      +ray_kwargs.ray_init.runtime_env.env_vars.FI_PROVIDER=""
  '

echo "=========================================="
echo "Smoke test finished."
echo "=========================================="
