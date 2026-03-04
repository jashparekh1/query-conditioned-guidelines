#!/bin/bash
#SBATCH --account=bfgx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=debug-10q
#SBATCH --chdir=/projects/bfgx/jparekh/query-conditioned-guidelines
#SBATCH --output=experiments/logs/debug_10q_%j.out
#SBATCH --error=experiments/logs/debug_10q_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

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

echo "Debug 10q: 7B planner + 14B-AWQ solver, full output"
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
    pip install -q math-verify 2>/dev/null || true
    export PYTHONPATH="/workspace/query-conditioned-guidelines:${PYTHONPATH}"
    python -m experiments.debug_trajectories \
      --n 10 \
      --rollouts 10 \
      --model Qwen/Qwen2.5-7B-Instruct \
      --solver Qwen/Qwen2.5-14B-Instruct-AWQ \
      --data experiments/data/numinamath_30k/train.parquet \
      --seed 123 \
      --out experiments/logs/debug_trajectories_full.json
  '

echo "Done."
