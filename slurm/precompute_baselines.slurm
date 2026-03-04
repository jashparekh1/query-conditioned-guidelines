#!/bin/bash
#SBATCH --account=bfgx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=baselines
#SBATCH --chdir=/projects/bfgx/jparekh/query-conditioned-guidelines
#SBATCH --output=experiments/logs/precompute_baselines_%j.out
#SBATCH --error=experiments/logs/precompute_baselines_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G

# Pre-compute solver baselines (no guideline) for counterfactual reward.
# Only needs 1 GPU for the 14B-AWQ solver. ~30k problems, ~1-2 hours.
# Output: experiments/data/numinamath_30k/baselines.json
# Resume-safe: re-run the same job to pick up where it left off.

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
echo "Pre-computing solver baselines (no guideline)"
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
  --env HF_HOME=/tmp/hf_cache \
  --env HF_DATASETS_CACHE=/tmp/hf_cache \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env TMPDIR=/tmp \
  --env VLLM_USE_V1=0 \
  "$SIF" \
  bash -c '
    cd /workspace/query-conditioned-guidelines
    export PYTHONPATH="/workspace/query-conditioned-guidelines:/opt/verl:${PYTHONPATH}"
    pip install -q math-verify || true
    python -m experiments.precompute_baselines \
      --input experiments/data/numinamath_30k/train.parquet \
      --output experiments/data/numinamath_30k/baselines.json \
      --model Qwen/Qwen2.5-14B-Instruct-AWQ \
      --batch_size 32 \
      --gpu_memory_utilization 0.8 \
      --resume
  '

echo "=========================================="
echo "Done."
echo "=========================================="
