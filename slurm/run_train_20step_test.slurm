#!/bin/bash
#SBATCH --account=bfgx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=train-20step
#SBATCH --chdir=/projects/bfgx/jparekh/query-conditioned-guidelines
#SBATCH --output=experiments/logs/train_20step_%j.out
#SBATCH --error=experiments/logs/train_20step_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G

# 20-step smoke test via sbatch. Same as run_train_20step_test.sh but runs in batch.
# Optional: export WANDB_API_KEY=your_key before sbatch for wandb logging.
# Submit from repo root so --chdir finds the repo (or adjust --chdir to your path).

set -e
# Job runs in repo root via --chdir
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
  echo "Copy your built SIF to: $SIF_REPO"
  exit 1
fi

# Ensure WORK_NVME exists and is writable (bound as /nvme for HF cache, apptainer cache)
if ! mkdir -p "$WORK_NVME/hf_cache" "$WORK_NVME/apptainer_cache"; then
  echo "ERROR: Cannot create $WORK_NVME (or subdirs). Check that /work/nvme/bfgx exists and is writable on this node."
  exit 1
fi
if ! touch "$WORK_NVME/hf_cache/.write_test" 2>/dev/null; then
  echo "ERROR: $WORK_NVME is not writable. Cannot use /nvme for HF cache."
  exit 1
fi
rm -f "$WORK_NVME/hf_cache/.write_test"

echo "=========================================="
echo "20-step train smoke test (sbatch)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "SIF: $SIF"
echo "=========================================="

# Avoid Apptainer warning: do not forward host HF_* into container
unset HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE 2>/dev/null || true

apptainer exec --nv \
  --bind "/projects/bfgx/jparekh:/workspace" \
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
    source /workspace/setup_verl_apptainer.sh
    cd /workspace/query-conditioned-guidelines
    pip install -q math-verify || { echo "ERROR: pip install math-verify failed"; exit 1; }
    bash experiments/run_train_20step_test.sh
  '

echo "=========================================="
echo "Job finished."
echo "=========================================="
