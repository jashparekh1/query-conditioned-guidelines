#!/bin/bash
#SBATCH --account=bfgx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=prm-corr
#SBATCH --chdir=/projects/bfgx/jparekh/query-conditioned-guidelines
#SBATCH --output=experiments/logs/prm_correlation_%j.out
#SBATCH --error=experiments/logs/prm_correlation_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# 10q x 10r: generate trajectories, PRM score, then compute correlation(min_PRM, correctness)

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

echo "PRM correlation experiment: 10q x 10r"
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

    OUT_JSON="experiments/logs/prm_correlation_10q10r.json"
    PRM_JSON="experiments/logs/prm_correlation_10q10r_scores.json"

    # Step 1: Generate 10q x 10r trajectories
    echo "=== Step 1: Generating 10q x 10r trajectories ==="
    python -m experiments.debug_trajectories \
      --n 10 \
      --rollouts 10 \
      --model Qwen/Qwen3-4B-Instruct-2507 \
      --solver Qwen/Qwen2.5-14B-Instruct-AWQ \
      --data experiments/data/dapomath_30k/train.parquet \
      --temperature 0.8 \
      --seed 99 \
      --out "$OUT_JSON"

    # Step 2: PRM scoring
    echo ""
    echo "=== Step 2: PRM scoring ==="
    python -m experiments.score_with_prm \
      --input "$OUT_JSON" \
      --model Qwen/Qwen2.5-Math-PRM-7B \
      --out "$PRM_JSON"

    # Step 3: Correlation analysis
    echo ""
    echo "=== Step 3: Correlation(min_PRM, correctness) ==="
    python -m experiments.analyze_prm_correlation --input "$PRM_JSON"
  '

echo "Done."
