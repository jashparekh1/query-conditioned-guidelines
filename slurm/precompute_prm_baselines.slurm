#!/bin/bash
#SBATCH --account=bfgx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=prm-base
#SBATCH --chdir=/projects/bfgx/jparekh/query-conditioned-guidelines
#SBATCH --output=experiments/logs/prm_baselines_%j.out
#SBATCH --error=experiments/logs/prm_baselines_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Pre-compute PRM baselines: min(PRM) for solver WITHOUT planner on 3k DAPO-Math questions.
# Phase 1: Solver inference (14B-AWQ, ~30min for 3k questions)
# Phase 2: PRM scoring (7B, ~60min for 3k questions)

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

echo "PRM baselines: 3000 questions, solver without guidelines + PRM scoring"
echo "Job ID: $SLURM_JOB_ID"

apptainer exec --nv \
  --bind "/projects/bfgx/jparekh:/workspace" \
  --bind "$WORK_NVME:/nvme" \
  --env HF_HOME=/tmp/hf_cache \
  --env HF_HUB_CACHE=/tmp/hf_cache \
  --env TRANSFORMERS_CACHE=/tmp/hf_cache \
  --env HF_DATASETS_CACHE=/tmp/hf_cache \
  --env XDG_CACHE_HOME=/tmp/xdg_cache \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env VLLM_USE_V1=0 \
  "$SIF" \
  bash -c '
    cd /workspace/query-conditioned-guidelines
    export PYTHONPATH="/workspace/query-conditioned-guidelines:${PYTHONPATH}"
    python -m experiments.precompute_prm_baselines \
      --input experiments/data/dapomath_30k/train.parquet \
      --output experiments/data/dapomath_30k/prm_baselines.json \
      --solver Qwen/Qwen2.5-14B-Instruct-AWQ \
      --prm Qwen/Qwen2.5-Math-PRM-7B \
      --n 3000 \
      --seed 42 \
      --batch_size 32 \
      --resume
  '

echo "Done."
