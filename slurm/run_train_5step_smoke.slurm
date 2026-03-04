#!/bin/bash
#SBATCH --account=bfgx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=smoke-v2
#SBATCH --chdir=/projects/bfgx/jparekh/query-conditioned-guidelines
#SBATCH --output=experiments/logs/train_5step_smoke_%j.out
#SBATCH --error=experiments/logs/train_5step_smoke_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G

# 5-step smoke test for v2 config: validates 14B-GPTQ solver loads, GRPO runs, rewards come back.
# Should finish in <30 min if everything works. Check the .out file for errors.

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

echo "=========================================="
echo "5-step smoke test: v2 config + 14B-AWQ solver"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "SIF: $SIF"
echo "=========================================="

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
  --env NCCL_NET_PLUGIN="" \
  --env VERL_GUIDELINES_USE_MATH_VERIFY="${VERL_GUIDELINES_USE_MATH_VERIFY:-1}" \
  --env VERL_GUIDELINES_LOG_DIR="/workspace/query-conditioned-guidelines/experiments/logs/solver_debug" \
  "$SIF" \
  bash -c '
    source /workspace/setup_verl_apptainer.sh 2>/dev/null || true
    cd /workspace/query-conditioned-guidelines
    pip install -q math-verify || { echo "ERROR: pip install math-verify failed"; exit 1; }
    echo "=== Starting 5-step training ==="
    export VERL_EXPERIMENT_SUFFIX="-v2-smoke"
    bash experiments/run_train.sh \
      trainer.total_training_steps=5 \
      trainer.experiment_name=smoke-v2-14bgptq \
      trainer.project_name=jparekh-test \
      trainer.save_freq=999 \
      trainer.test_freq=999 \
      trainer.logger=console
  '

echo "=========================================="
echo "Smoke test finished. Check .out for errors."
echo "=========================================="
