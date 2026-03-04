#!/bin/bash
#SBATCH --account=bfgx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=train-300-v6
#SBATCH --chdir=/projects/bfgx/jparekh/query-conditioned-guidelines
#SBATCH --output=experiments/logs/train_300step_v6_%j.out
#SBATCH --error=experiments/logs/train_300step_v6_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G

# v5: 7B planner (up from 3B), new specific-prompt, counterfactual reward, 14B-AWQ solver.
# batch=60, rollout_n=32 (1920 samples/step), lr=1e-6, KL loss only.
# IMPORTANT: Requires regenerated train.parquet with new planner prompt:
#   apptainer exec --nv ... bash -c 'cd /workspace/query-conditioned-guidelines && python -m experiments.prepare_numinamath --output_dir experiments/data/numinamath_30k'
# Wandb run: numinamath30k-7b-300steps-v6. Slurm logs: experiments/logs/train_300step_v6_<jobid>.out|err

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
echo "300-step v6: 7B planner, new prompt, counterfactual reward, 14B-AWQ solver"
echo "batch=60, rollout_n=32, lr=1e-6"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "SIF: $SIF"
echo "=========================================="

# Clean start (default): wipe NVMe checkpoint so verl resume_mode=auto starts from scratch
CKPT_NAME="numinamath30k-7b-300steps-v6"
if [[ "${VERL_CLEAN_START:-1}" =~ ^(1|true|yes)$ ]]; then
  echo "VERL_CLEAN_START=1: Removing NVMe checkpoint to start fresh..."
  rm -rf "$WORK_NVME/checkpoints/$CKPT_NAME"
fi

# Resume: if project has a checkpoint and NVMe doesn't, copy project -> NVMe
PROJECT_CKPT="$PROJECT_ROOT/checkpoints/query-conditioned-guidelines/$CKPT_NAME"
NVME_CKPT="$WORK_NVME/checkpoints/$CKPT_NAME"
if [[ -f "$PROJECT_CKPT/latest_checkpointed_iteration.txt" ]]; then
  STEP=$(cat "$PROJECT_CKPT/latest_checkpointed_iteration.txt" | tr -d '[:space:]')
  if [[ -n "$STEP" && -d "$PROJECT_CKPT/global_step_$STEP" ]]; then
    if [[ ! -f "$NVME_CKPT/latest_checkpointed_iteration.txt" ]] || [[ "$(cat "$NVME_CKPT/latest_checkpointed_iteration.txt" | tr -d '[:space:]')" -lt "$STEP" ]]; then
      echo "Resuming: copying checkpoint step $STEP from project to NVMe..."
      mkdir -p "$NVME_CKPT"
      cp -r "$PROJECT_CKPT/global_step_$STEP" "$NVME_CKPT/"
      cp "$PROJECT_CKPT/latest_checkpointed_iteration.txt" "$NVME_CKPT/"
      echo "Done. Will resume from step $STEP."
    fi
  fi
fi

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
  --env NCCL_NET_PLUGIN="" \
  --env VERL_GUIDELINES_USE_MATH_VERIFY="${VERL_GUIDELINES_USE_MATH_VERIFY:-1}" \
  --env VERL_DIST_TIMEOUT_SECONDS="${VERL_DIST_TIMEOUT_SECONDS:-1800}" \
  --env VERL_CHECKPOINT_DIR="/nvme/checkpoints/$CKPT_NAME" \
  --env VERL_SAVE_FREQ=100 \
  --env VERL_GUIDELINES_LOG_DIR="${VERL_GUIDELINES_LOG_DIR:-/workspace/query-conditioned-guidelines/experiments/logs/solver_debug_v6}" \
  "$SIF" \
  bash -c '
    source /workspace/setup_verl_apptainer.sh 2>/dev/null || true
    cd /workspace/query-conditioned-guidelines
    pip install -q math-verify || { echo "ERROR: pip install math-verify failed"; exit 1; }
    # Force NCCL to use shared memory (intra-node) instead of AWS Libfabric
    export NCCL_NET_PLUGIN=""
    export NCCL_PLUGIN_P2P=""
    mkdir -p /nvme/checkpoints/'"$CKPT_NAME"'
    export VERL_EXPERIMENT_SUFFIX="-v6"
    export VERL_ROLLOUT_N=32
    export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
    export VERL_VAL_FILE="/workspace/query-conditioned-guidelines/experiments/data/math500_eval/test.parquet"
    bash experiments/run_train.sh
  ' || TRAIN_EXIT=$?

# Copy last checkpoint from NVMe to project when job ends
NVME_CKPT="$WORK_NVME/checkpoints/$CKPT_NAME"
PROJECT_CKPT="$PROJECT_ROOT/checkpoints/query-conditioned-guidelines/$CKPT_NAME"
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
