#!/bin/bash
#SBATCH --account=bfgx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=probe-dapo
#SBATCH --chdir=/projects/bfgx/jparekh/query-conditioned-guidelines
#SBATCH --output=experiments/logs/probe_dapomath_%j.out
#SBATCH --error=experiments/logs/probe_dapomath_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Probe DAPO-Math-17k: run 14B-AWQ solver (no guideline) on 200 questions.
# Reports accuracy to decide if this dataset is the right difficulty.
# Also runs 7B solver for comparison.

set -e
PROJECT_ROOT="/projects/bfgx/jparekh/query-conditioned-guidelines"
cd "$PROJECT_ROOT"
mkdir -p experiments/logs

USER="${USER:-jparekh}"
WORK_NVME="/work/nvme/bfgx/$USER"
SIF_REPO="$PROJECT_ROOT/containers/verl-vllm-24.04.sif"
SIF="${SIF_REPO:-$WORK_NVME/verl-vllm-24.04.sif}"

echo "Probing DAPO-Math-17k solver accuracy"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"

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

    echo "=== 14B-AWQ solver ==="
    python -m experiments.probe_dapomath \
      --n 200 \
      --solver Qwen/Qwen2.5-14B-Instruct-AWQ \
      --out experiments/logs/probe_dapomath_14b.json

    echo "=== 7B solver ==="
    python -m experiments.probe_dapomath \
      --n 200 \
      --solver Qwen/Qwen2.5-7B-Instruct \
      --out experiments/logs/probe_dapomath_7b.json
  '

echo "Done. Results in experiments/logs/probe_dapomath_14b.json and probe_dapomath_7b.json"
