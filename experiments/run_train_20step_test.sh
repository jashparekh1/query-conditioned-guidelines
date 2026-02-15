#!/usr/bin/env bash
# 20-step smoke test: same as full training but stops at 20 steps.
# Use this to verify the pipeline (e.g. on a 2hr node) before running the full 300-step job via sbatch.
# Config: 1 node × 4 GPUs, same as previous runs.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Batch sizes must satisfy: (train_batch_size * rollout.n) % n_gpus == 0 (n_gpus=3).
# 30*4=120 % 3=0; ppo_mini_batch_size must be <= train_batch_size and divisible by ppo_micro_batch_size_per_gpu (8).
# Match run_train.sh naming: -mathverify suffix when VERL_GUIDELINES_USE_MATH_VERIFY is on
[[ "${VERL_GUIDELINES_USE_MATH_VERIFY:-1}" =~ ^(1|true|yes)$ ]] && VERL_EXPERIMENT_SUFFIX="-mathverify" || VERL_EXPERIMENT_SUFFIX=""
bash experiments/run_train.sh \
  data.train_batch_size=30 \
  actor_rollout_ref.actor.ppo_mini_batch_size=24 \
  trainer.total_training_steps=20 \
  trainer.experiment_name=numinamath30k-3b-20step-test${VERL_EXPERIMENT_SUFFIX} \
  trainer.project_name=jparekh-test \
  trainer.save_freq=10 \
  trainer.test_freq=20 \
  "$@" 2>&1 | tee experiments/logs/train_20step_test_$(date +%Y%m%d_%H%M%S).log
