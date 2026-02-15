# Training config reference (full 300-step run)

**Algorithm: GRPO** (Group Relative Policy Optimization — no critic; advantages from group baseline. Policy update is PPO-style; config keys still use `ppo_*`.)

## GPU allocation (1 node, 4 GPUs)

| GPU(s) | Role | What runs there |
|--------|------|-----------------|
| **Pool GPU 0** | Rollout vLLM | Planner sampling: generate 120 responses (30 prompts × 4 rollouts) per step. `tensor_model_parallel_size=1` → 1 GPU. |
| **Pool GPU 1–2** | Actor + Ref | FSDP actor (policy) + reference model for log-probs / KL. Share these 2 GPUs. |
| **Pool GPU 3** | **TaskRunner (solver)** | Reward: load solver vLLM (same 3B model), run 120 solver calls per step (guideline + question → answer), then grade. `num_gpus=1` for this Ray actor. |

- **Ray:** `trainer.n_gpus_per_node=3` → resource pool gets 3 GPUs (rollout + actor + ref). TaskRunner is a separate Ray actor with `num_gpus=1`, so it gets the 4th GPU. Total = 4.

---

## Key parameters (from `run_train.sh`)

### Data & sequence
| Param | Value | Meaning |
|-------|--------|---------|
| `data.train_batch_size` | 30 | Prompts per step (per data loader). |
| `data.max_prompt_length` | 3072 | Max prompt tokens. |
| `data.max_response_length` | 3072 | Max response tokens (planner + solver). |
| `data.filter_overlong_prompts` | True | Drop prompts > 3072. |
| `data.truncation` | error | Fail on overlong instead of truncating. |

### Rollouts (planner)
| Param | Value | Meaning |
|-------|--------|---------|
| `actor_rollout_ref.rollout.n` | 4 | Rollouts per prompt (samples per prompt). |
| **Effective batch** | **30 × 4 = 120** | Total responses per step (real_train_batch_size). |
| `actor_rollout_ref.rollout.name` | vllm | Use vLLM for rollout. |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | 1 | Rollout vLLM on 1 GPU. |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.4 | vLLM memory fraction. |
| `actor_rollout_ref.rollout.enable_chunked_prefill` | False | No chunked prefill. |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` | 8 | Micro-batch for log-prob re-computation. |

### Actor (policy) — GRPO / PPO-style update
| Param | Value | Meaning |
|-------|--------|---------|
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 24 | Mini-batch size for policy update (must be ≤ train_batch_size, divisible by micro). |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | 8 | Micro-batch per GPU for actor backward. |
| `actor_rollout_ref.actor.optim.lr` | 1e-6 | Learning rate. |
| `actor_rollout_ref.actor.use_kl_loss` | True | Add KL penalty to policy loss. |
| `actor_rollout_ref.actor.kl_loss_coef` | 0.001 | KL coefficient. |
| `actor_rollout_ref.actor.kl_loss_type` | low_var_kl | KL formulation. |
| `actor_rollout_ref.actor.entropy_coeff` | 0 | No entropy bonus. |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | True | Recompute activations to save memory. |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | False | No CPU offload for actor. |

### Reference model
| Param | Value | Meaning |
|-------|--------|---------|
| `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | 4 | Micro-batch for ref log-probs. |
| `actor_rollout_ref.ref.fsdp_config.param_offload` | True | Ref model params offloaded to CPU to save GPU memory. |

### Reward (solver)
| Param | Value | Meaning |
|-------|--------|---------|
| Solver model | Same as planner (Qwen2.5-3B-Instruct) | From `VERL_GUIDELINES_SOLVER_MODEL`. |
| Solver GPU | 1 (TaskRunner) | 120 solver calls per step on that GPU; legacy vLLM engine (`VLLM_USE_V1=0`). |
| **Reward** | **Accuracy only** | Reward = 1.0 if solver answer is correct, 0.0 otherwise. No format or length terms. |
| **Accuracy grading** | **math_verify** (when `VERL_GUIDELINES_USE_MATH_VERIFY=1`) | Correctness uses math_verify for math equivalence (e.g. 0.5 == 1/2). Slurm/container scripts run `pip install -q math-verify` before training. |
| **Checkpoint dir (mathverify run)** | `checkpoints/query-conditioned-guidelines/numinamath30k-3b-300steps-mathverify/` | Explicit `trainer.default_local_dir` so the 300-step mathverify run never overwrites the first (non-mathverify) run. First run uses `.../numinamath30k-3b-300steps/`. |

### Algorithm & trainer
| Param | Value | Meaning |
|-------|--------|---------|
| `algorithm.adv_estimator` | **grpo** | **GRPO** (group-relative advantages, no critic). |
| `algorithm.use_kl_in_reward` | True | KL penalty in reward (reward = score - beta*KL); keeps policy near ref. |
| `trainer.total_training_steps` | 300 | Target steps. |
| `trainer.total_epochs` | 10 | Epoch cap. |
| `trainer.save_freq` | 25 | Checkpoint every 25 steps. |
| `trainer.test_freq` | 50 | Validation every 50 steps. |
| `trainer.n_gpus_per_node` | 3 | GPUs in training pool (see above). |
| `trainer.nnodes` | 1 | Single node. |

### Model
| Param | Value |
|-------|--------|
| `actor_rollout_ref.model.path` | Qwen/Qwen2.5-3B-Instruct (planner and solver). |

---

## Why ~4 steps in 3 hours?

Per step you do:

1. **Rollout:** 120 generations (30 × 4) at up to 3072 tokens each on 1 GPU.
2. **Ref log-probs:** Over the same 120 responses (ref on 2 GPUs).
3. **Reward:** 120 solver runs (each: guideline + question → full solution, then grade) on 1 GPU — same 3B model, long sequences.
4. **Actor update:** GRPO policy update on 120 samples (PPO-style loss, mini-batches of 24).

The main cost is usually **reward**: 120 solver calls at 3072 max length on a single GPU, with no batching across the 120 in the current guidelines reward path, so it can be effectively 120 sequential or small-batch inferences per step. Rollout at 120 × 3072 on one GPU is also heavy. So a few steps per hour with this setup is expected; to speed up you’d look at smaller `train_batch_size` or `rollout.n`, lower max lengths, or batching/throughput improvements in the solver reward path.
