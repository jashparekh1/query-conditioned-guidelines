#!/usr/bin/env bash
# Allocate one node and run the 20-step train interactively (output streams to your terminal).
#
# Usage (from login node; script will srun to get a node):
#   ./slurm/run_train_20step_interactive.sh
#
# Usage (already on a compute node, e.g. inside srun --pty bash; no nested srun):
#   export RUN_IN_CURRENT_ALLOCATION=1
#   ./slurm/run_train_20step_interactive.sh
#
# Optional: export WANDB_API_KEY=your_key

set -e
REPO_ROOT="/projects/bfgx/jparekh/query-conditioned-guidelines"
cd "$REPO_ROOT"
mkdir -p experiments/logs

USER="${USER:-jparekh}"
WORK_NVME="/work/nvme/bfgx/$USER"
# Prefer SIF in repo so we can run with a shared image; fallback to WORK_NVME
SIF_REPO="$REPO_ROOT/containers/verl-vllm-24.04.sif"
if [[ -f "$SIF_REPO" ]]; then
  SIF="$SIF_REPO"
else
  SIF="$WORK_NVME/verl-vllm-24.04.sif"
fi

if [[ ! -f "$SIF" ]]; then
  echo "ERROR: SIF not found. Tried: $SIF_REPO and $WORK_NVME/verl-vllm-24.04.sif"
  echo "Copy your built SIF to: $SIF_REPO"
  exit 1
fi

echo "=========================================="
echo "20-step train (interactive)"
echo "=========================================="
echo "SIF: $SIF"
if [[ -n "${RUN_IN_CURRENT_ALLOCATION:-}" ]]; then
  echo "Mode: run in current allocation (no srun)"
else
  echo "Mode: srun to allocate node, then run container"
fi
echo "=========================================="

# Avoid Apptainer warning: do not forward host HF_* into container
unset HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE 2>/dev/null || true

_run_container() {
  cd "$REPO_ROOT"
  mkdir -p "$WORK_NVME/hf_cache" "$WORK_NVME/apptainer_cache" || { echo "ERROR: Cannot create $WORK_NVME"; return 1; }
  touch "$WORK_NVME/hf_cache/.write_test" || { echo "ERROR: $WORK_NVME not writable"; return 1; }
  rm -f "$WORK_NVME/hf_cache/.write_test"
  unset HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE 2>/dev/null || true
  apptainer exec --nv \
    --bind '/projects/bfgx/jparekh:/workspace' \
    --bind "$WORK_NVME:/nvme" \
    --bind "$WORK_NVME/apptainer_cache:/root/.cache" \
    --env VERL_PATH=/opt/verl \
    --env VLLM_TORCH_DTYPE=bfloat16 \
    --env TORCH_DTYPE=bfloat16 \
    --env HF_HOME=/tmp/hf_cache \
    --env HF_DATASETS_CACHE=/tmp/hf_cache \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    --env PYTHONUNBUFFERED=1 \
    --env PYTHONNOUSERSITE=1 \
    --env TMPDIR=/tmp \
    --env RAY_TMPDIR=/tmp \
    --env VERL_GUIDELINES_USE_MATH_VERIFY="${VERL_GUIDELINES_USE_MATH_VERIFY:-1}" \
    "$SIF" \
    bash -c '
      source /workspace/setup_verl_apptainer.sh 2>/dev/null || true
      cd /workspace/query-conditioned-guidelines
      pip install -q math-verify || { echo "ERROR: pip install math-verify failed"; exit 1; }
      bash experiments/run_train_20step_test.sh
    '
}

if [[ -n "${RUN_IN_CURRENT_ALLOCATION:-}" ]]; then
  _run_container
else
  srun --account=bfgx-dtai-gh --partition=ghx4 \
    --nodes=1 --ntasks=1 --gres=gpu:nvidia_gh200_120gb:4 --cpus-per-task=32 --mem=200G \
    --time=2:00:00 \
    bash -c "export RUN_IN_CURRENT_ALLOCATION=1; cd '$REPO_ROOT' && exec bash '$REPO_ROOT/slurm/run_train_20step_interactive.sh'"
fi

echo "=========================================="
echo "Done."
echo "=========================================="
