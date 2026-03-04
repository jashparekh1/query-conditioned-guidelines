#!/bin/bash
#SBATCH --account=bfgx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=baselines-dapo
#SBATCH --chdir=/projects/bfgx/jparekh/query-conditioned-guidelines
#SBATCH --output=experiments/logs/baselines_dapomath_%j.out
#SBATCH --error=experiments/logs/baselines_dapomath_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G

# Step 1: prepare DAPO-Math-30k parquet -> experiments/data/dapomath_30k/
# Step 2: precompute 14B-AWQ solver baselines in parallel across 2 GPUs
#   GPU 0: shard0 -> baselines_shard0.json (indices 0..14999)
#   GPU 1: shard1 -> baselines_shard1.json (indices 15000..29999)
# Step 3: merge shards -> baselines.json
# Resume-safe: re-run to continue from where it left off.

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

mkdir -p "$WORK_NVME/hf_cache" "$WORK_NVME/apptainer_cache" 2>/dev/null || true

echo "=========================================="
echo "DAPO-Math-30k: prepare parquet + precompute 14B baselines (2 GPUs)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "=========================================="

unset HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE 2>/dev/null || true

APPTAINER_ARGS="--nv
  --bind /projects/bfgx/jparekh:/workspace
  --bind $WORK_NVME:/nvme
  --bind $WORK_NVME/apptainer_cache:/root/.cache
  --env HF_HOME=/tmp/hf_cache
  --env HF_HUB_CACHE=/tmp/hf_cache
  --env TRANSFORMERS_CACHE=/tmp/hf_cache
  --env HF_DATASETS_CACHE=/tmp/hf_cache
  --env XDG_CACHE_HOME=/tmp/xdg_cache
  --env HF_TOKEN=${HF_TOKEN:-}
  --env TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real
  --env PYTHONUNBUFFERED=1
  --env PYTHONNOUSERSITE=1
  --env TMPDIR=/tmp
  --env VLLM_USE_V1=0"

DATA_OUT="$PROJECT_ROOT/experiments/data/dapomath_30k"

# Step 1: prepare parquet (single-threaded, quick)
if [[ -f "$DATA_OUT/train.parquet" ]]; then
  echo "train.parquet already exists, skipping prepare step."
else
  echo "=== Step 1: Preparing DAPO-Math-30k parquet ==="
  apptainer exec $APPTAINER_ARGS "$SIF" bash -c '
    cd /workspace/query-conditioned-guidelines
    export PYTHONPATH="/workspace/query-conditioned-guidelines:/opt/verl:${PYTHONPATH}"
    pip install -q math-verify || true
    python -m experiments.prepare_dapomath \
      --output_dir experiments/data/dapomath_30k \
      --subsample 30000 \
      --seed 42
  '
  echo "Done. $(ls -lh $DATA_OUT/train.parquet)"
fi

# Determine mid-point for sharding
N_TRAIN=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$DATA_OUT/train.parquet')))")
MID=$(( N_TRAIN / 2 ))
echo "Train size: $N_TRAIN, shard split: [0,$MID) and [$MID,$N_TRAIN)"

# Step 2: run both shards in parallel, one per GPU
echo "=== Step 2: Precomputing baselines in parallel (GPU 0: shard0, GPU 1: shard1) ==="

apptainer exec $APPTAINER_ARGS --env CUDA_VISIBLE_DEVICES=0 "$SIF" bash -c '
  cd /workspace/query-conditioned-guidelines
  export PYTHONPATH="/workspace/query-conditioned-guidelines:/opt/verl:${PYTHONPATH}"
  pip install -q math-verify || true
  python -m experiments.precompute_baselines \
    --input experiments/data/dapomath_30k/train.parquet \
    --output experiments/data/dapomath_30k/baselines_shard0.json \
    --model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --batch_size 64 \
    --gpu_memory_utilization 0.85 \
    --start_idx 0 \
    --end_idx '"$MID"' \
    --resume
' &
PID0=$!

apptainer exec $APPTAINER_ARGS --env CUDA_VISIBLE_DEVICES=1 "$SIF" bash -c '
  cd /workspace/query-conditioned-guidelines
  export PYTHONPATH="/workspace/query-conditioned-guidelines:/opt/verl:${PYTHONPATH}"
  pip install -q math-verify || true
  python -m experiments.precompute_baselines \
    --input experiments/data/dapomath_30k/train.parquet \
    --output experiments/data/dapomath_30k/baselines_shard1.json \
    --model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --batch_size 64 \
    --gpu_memory_utilization 0.85 \
    --start_idx '"$MID"' \
    --end_idx '"$N_TRAIN"' \
    --resume
' &
PID1=$!

echo "Shard 0 PID: $PID0 | Shard 1 PID: $PID1"
wait $PID0 && echo "Shard 0 done" || { echo "ERROR: Shard 0 failed"; kill $PID1 2>/dev/null; exit 1; }
wait $PID1 && echo "Shard 1 done" || { echo "ERROR: Shard 1 failed"; exit 1; }

# Step 3: merge shards -> baselines.json
echo "=== Step 3: Merging shards ==="
python3 -c "
import json
merged = {}
for path in ['$DATA_OUT/baselines_shard0.json', '$DATA_OUT/baselines_shard1.json']:
    with open(path) as f:
        merged.update(json.load(f))
with open('$DATA_OUT/baselines.json', 'w') as f:
    json.dump({str(k): v for k, v in sorted(merged.items(), key=lambda x: int(x[0]))}, f)
total = len(merged)
correct = sum(v for v in merged.values())
print(f'Merged {total} baselines, overall 14B accuracy (no guideline): {correct/total:.3f}')
"

echo "=========================================="
echo "Done. baselines.json written to $DATA_OUT/"
echo "=========================================="
