#!/bin/bash
#SBATCH --account=bfgx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=smoke-v5
#SBATCH --chdir=/projects/bfgx/jparekh/query-conditioned-guidelines
#SBATCH --output=experiments/logs/train_5step_smoke_v5_%j.out
#SBATCH --error=experiments/logs/train_5step_smoke_v5_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G

# v5 smoke test: 7B planner, new prompt, counterfactual reward, 5 steps only.

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

mkdir -p "$WORK_NVME/hf_cache" "$WORK_NVME/apptainer_cache" 2>/dev/null || true

echo "=========================================="
echo "5-step smoke test v5: 7B planner, new prompt, counterfactual reward"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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
  --env VERL_DIST_TIMEOUT_SECONDS="${VERL_DIST_TIMEOUT_SECONDS:-1800}" \
  --env VERL_GUIDELINES_LOG_DIR="/workspace/query-conditioned-guidelines/experiments/logs/solver_debug_v5_smoke" \
  "$SIF" \
  bash -c '
    source /workspace/setup_verl_apptainer.sh 2>/dev/null || true
    cd /workspace/query-conditioned-guidelines
    pip install -q math-verify || { echo "ERROR: pip install math-verify failed"; exit 1; }
    # Force NCCL to use shared memory (intra-node) instead of AWS Libfabric
    export NCCL_NET_PLUGIN=""
    export NCCL_PLUGIN_P2P=""
    export VERL_EXPERIMENT_SUFFIX="-v5-smoke"
    export VERL_ROLLOUT_N=4
    export VERL_TOTAL_STEPS=5
    export VERL_SAVE_FREQ=5
    export VERL_TEST_FREQ=5
    export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
    bash experiments/run_train.sh \
      trainer.logger=console \
      trainer.project_name=jparekh-test \
      trainer.experiment_name=smoke-v5-7b
  '

echo "=========================================="
echo "Smoke test done."
echo "=========================================="
