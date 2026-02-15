#!/bin/bash
#SBATCH --account=bfgx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=train-300-mv
#SBATCH --chdir=/projects/bfgx/jparekh/query-conditioned-guidelines
#SBATCH --output=experiments/logs/train_300step_mathverify_%j.out
#SBATCH --error=experiments/logs/train_300step_mathverify_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G

# Full 300-step training with math_verify reward: 3072 length, checkpoint every 50 steps to NVMe, 48hr limit.
# Uses VERL_GUIDELINES_USE_MATH_VERIFY=1; checkpoints go to /nvme/... (bind = $WORK_NVME = /work/nvme/bfgx/$USER).
# Checkpoints: /work/nvme/bfgx/$USER/checkpoints/numinamath30k-3b-300steps-mathverify/ (node-local; copy to project when done if needed).
# Wandb run: numinamath30k-3b-300steps-mathverify. Slurm logs: experiments/logs/train_300step_mathverify_<jobid>.out|err
# Optional: export WANDB_API_KEY=your_key before sbatch. Submit from repo root.

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
if ! touch "$WORK_NVME/hf_cache/.write_test" 2>/dev/null; then
  echo "ERROR: $WORK_NVME is not writable."
  exit 1
fi
rm -f "$WORK_NVME/hf_cache/.write_test"

echo "=========================================="
echo "300-step full training (48hr, save every 50 to NVMe)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "SIF: $SIF"
echo "=========================================="

unset HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE 2>/dev/null || true

TRAIN_EXIT=0
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
  --env VERL_DIST_TIMEOUT_SECONDS="${VERL_DIST_TIMEOUT_SECONDS:-1800}" \
  --env VERL_CHECKPOINT_DIR=/nvme/checkpoints/numinamath30k-3b-300steps-mathverify \
  --env VERL_SAVE_FREQ=50 \
  "$SIF" \
  bash -c '
    source /workspace/setup_verl_apptainer.sh 2>/dev/null || true
    cd /workspace/query-conditioned-guidelines
    pip install -q math-verify || { echo "ERROR: pip install math-verify failed"; exit 1; }
    mkdir -p /nvme/checkpoints/numinamath30k-3b-300steps-mathverify
    bash experiments/run_train.sh
  ' || TRAIN_EXIT=$?

# Copy last checkpoint from NVMe to project when job ends (success, failure, or time limit)
NVME_CKPT="$WORK_NVME/checkpoints/numinamath30k-3b-300steps-mathverify"
PROJECT_CKPT="$PROJECT_ROOT/checkpoints/query-conditioned-guidelines/numinamath30k-3b-300steps-mathverify"
if [[ -f "$NVME_CKPT/latest_checkpointed_iteration.txt" ]]; then
  STEP=$(cat "$NVME_CKPT/latest_checkpointed_iteration.txt" | tr -d '[:space:]')
  if [[ -n "$STEP" && -d "$NVME_CKPT/global_step_$STEP" ]]; then
    echo "=========================================="
    echo "Copying last checkpoint (step $STEP) from NVMe to project..."
    mkdir -p "$PROJECT_CKPT"
    cp -r "$NVME_CKPT/global_step_$STEP" "$PROJECT_CKPT/"
    cp "$NVME_CKPT/latest_checkpointed_iteration.txt" "$PROJECT_CKPT/"
    echo "Done. Resumable from $PROJECT_CKPT (step $STEP)."
    echo "=========================================="
  fi
fi

echo "=========================================="
echo "Job finished."
echo "=========================================="
exit $TRAIN_EXIT
