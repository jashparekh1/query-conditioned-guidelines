# 🚀 Quick Start for GRPO Training with VERL

## 1. Install Dependencies

Run the installation script:
bash scripts/install_vllm_sglang_mcore.sh

Then install the package in editable mode:
pip install -e .

---

## 2. Prepare Model and Data

Download your model and dataset into your preferred directory.

Preprocess the GSM8K dataset:
python examples/data_preprocess/gsm8k.py \
  --local_save_dir /your/data/path

---

## 3. Launch Training

The following command was tested on 4× A6000 GPUs and runs smoothly.

HOME=/shared/data2/jiashuo5
export WANDB_API_KEY=YOUR_WANDB_KEY

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/verl/data/gsm8k_processed/train.parquet \
  data.val_files=$HOME/verl/data/gsm8k_processed/test.parquet \
  data.train_batch_size=64 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$HOME/pretrain_models/Qwen2.5-3B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
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
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["wandb"]' \
  +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY=$WANDB_API_KEY \
  trainer.project_name='546-nlp-gsm8k-grpo' \
  trainer.experiment_name='gsm8k-grpo-qwen2.5-3b-instruct' \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=200 \
  trainer.test_freq=200 \
  trainer.total_epochs=5 \
  2>&1 | tee verl_training.log

---

## ✅ Notes

- Tested environment: 4× NVIDIA A6000 GPUs
- Logging: Weights & Biases (W&B) via WANDB_API_KEY
- Output logs saved to: verl_training.log
