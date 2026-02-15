#!/usr/bin/env bash
# Train guilder (planner) on NuminaMath-CoT 30k for 300 steps.
# - Model: Qwen2.5-3B-Instruct (planner and solver same model)
# - Sequence length: 3072
# - Checkpoint: default every 25 steps; set VERL_SAVE_FREQ=50 and VERL_CHECKPOINT_DIR for 300-step (e.g. NVMe).
# - Same prompts as eval (see experiments/prompts.py)
#
# Multi-node: set trainer.nnodes and trainer.n_gpus_per_node (total GPUs = nnodes * n_gpus_per_node).
#   Previous runs used 1 node, 4 GPUs. For more nodes, start a Ray cluster across nodes and pass
#   trainer.nnodes=2 trainer.n_gpus_per_node=4 (etc.) on the command line or below.
#
# API keys (see experiments/README.md):
#   - WANDB: set WANDB_API_KEY for logging (or disable with trainer.logger=console).
#   - HF: optional; only needed for gated/private models (e.g. HF_TOKEN or HUGGING_FACE_HUB_TOKEN).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Paths: set DATA_DIR to directory containing train.parquet from prepare_numinamath.py
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/experiments/data/numinamath_30k}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"

# Solver model for reward computation (same as planner for this experiment)
export VERL_GUIDELINES_SOLVER_MODEL="${MODEL_PATH}"
# Use math_verify for reward accuracy (math equivalence: 0.5 == 1/2). Requires: pip install math-verify
export VERL_GUIDELINES_USE_MATH_VERIFY="${VERL_GUIDELINES_USE_MATH_VERIFY:-1}"
# Experiment name suffix so wandb/logs distinguish math_verify vs mathruler runs
if [[ "${VERL_GUIDELINES_USE_MATH_VERIFY}" =~ ^(1|true|yes)$ ]]; then
  export VERL_EXPERIMENT_SUFFIX="-mathverify"
else
  export VERL_EXPERIMENT_SUFFIX=""
fi

# Optional: pass WANDB_API_KEY to Ray workers for wandb logging
export WANDB_API_KEY="${WANDB_API_KEY:-}"
# Checkpoint dir (default: under project). Set to /nvme/... in 300-step slurm to use node NVMe and avoid full /projects.
export VERL_CHECKPOINT_DIR="${VERL_CHECKPOINT_DIR:-checkpoints/query-conditioned-guidelines/numinamath30k-3b-300steps${VERL_EXPERIMENT_SUFFIX}}"
# Save frequency (default 25). Use 50 for 300-step to reduce checkpoint count and space.
export VERL_SAVE_FREQ="${VERL_SAVE_FREQ:-25}"

# Optional: verl/vllm from config if you use config/paths.sh
if [ -f config/paths.sh ]; then
  source config/paths.sh
fi
# In container: prefer image verl (/opt/verl), else /nvme/verl, so we don't pick up repo or paths.sh verl
if [[ -d /opt/verl ]]; then
  export VERL_PATH=/opt/verl
  export VLLM_PATH=
  # Force writable HF/datasets cache in container. Use /tmp (node-local, usually has space);
  # /nvme can be full on compute nodes, so avoid it for large caches.
  export HF_HOME=/tmp/hf_cache
  export HF_DATASETS_CACHE=/tmp/hf_cache
  export TRANSFORMERS_CACHE=/tmp/hf_cache
  export HUGGINGFACE_HUB_CACHE=/tmp/hf_cache
elif [[ -d /nvme/verl ]]; then
  export VERL_PATH=/nvme/verl
  export VLLM_PATH=
  export HF_HOME=/tmp/hf_cache
  export HF_DATASETS_CACHE=/tmp/hf_cache
  export TRANSFORMERS_CACHE=/tmp/hf_cache
  export HUGGINGFACE_HUB_CACHE=/tmp/hf_cache
fi
export PYTHONPATH="${PROJECT_ROOT}:${VERL_PATH:-$PROJECT_ROOT}:${VLLM_PATH:-}:${PYTHONPATH}"

echo "Training config: VERL_PATH=$VERL_PATH VLLM_PATH=${VLLM_PATH:-<container default>} DATA_DIR=$DATA_DIR"
echo "Train file: $DATA_DIR/train.parquet"

if [ ! -f "$DATA_DIR/train.parquet" ]; then
  echo "Missing $DATA_DIR/train.parquet. Run: python -m experiments.prepare_numinamath --output_dir $DATA_DIR"
  exit 1
fi

# 4 GPUs on node: 3 for training pool (rollout + actor + ref), 1 for TaskRunner (solver). n_gpus_per_node=3 so resource pool wants 3; TaskRunner takes the 4th.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size=30 \
  data.max_prompt_length=3072 \
  data.max_response_length=3072 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=24 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=True \
  trainer.critic_warmup=0 \
  trainer.logger=wandb \
  trainer.project_name=query-conditioned-guidelines \
  trainer.experiment_name=numinamath30k-3b-300steps${VERL_EXPERIMENT_SUFFIX} \
  trainer.default_local_dir="$VERL_CHECKPOINT_DIR" \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=3 \
  trainer.nnodes=1 \
  trainer.total_epochs=10 \
  trainer.total_training_steps=300 \
  trainer.save_freq="$VERL_SAVE_FREQ" \
  trainer.test_freq=50 \
  $([ -n "$WANDB_API_KEY" ] && echo "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY=$WANDB_API_KEY") \
  "$@" 2>&1 | tee experiments/logs/train_$(date +%Y%m%d_%H%M%S).log
