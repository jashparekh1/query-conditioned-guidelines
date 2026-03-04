# Query-Conditioned Guidelines for Math Reasoning

Train a small **planner** model to generate natural-language strategy guidelines that help a larger frozen **solver** solve math problems more accurately. The planner is trained via GRPO (Group Relative Policy Optimization) using [verl](https://github.com/volcengine/verl); the solver runs offline batch inference with [vLLM](https://github.com/vllm-project/vllm).

## Architecture

```
                  ┌───────────────┐
  question ──────>│  Planner (4B) │──── guideline (plain English strategy)
                  │  (trainable)  │
                  └───────────────┘
                          │
                          v
                  ┌───────────────┐
  question + ────>│  Solver (14B) │──── answer in \boxed{}
  guideline       │   (frozen)    │
                  └───────────────┘
                          │
                          v
                  ┌───────────────┐
                  │  math_verify  │──── reward: 1.0 if correct, 0.0 otherwise
                  └───────────────┘
```

- **Planner**: Qwen3-4B-Instruct-2507 (trainable). Outputs plain-English strategy only -- no math, no LaTeX, no `\boxed{}`.
- **Solver**: Qwen2.5-14B-Instruct-AWQ (frozen). Follows the guideline, reasons in `<think>` tags, puts final answer in `\boxed{}`.
- **Reward**: Binary accuracy via [math-verify](https://pypi.org/project/math-verify/) (handles equivalences like `0.5 == 1/2`). `R = 1.0` if the solver's answer matches ground truth, `0.0` otherwise. We have experimented with gain-beyond-baseline (counterfactual) rewards: `R = acc_with_guideline - acc_without_guideline`. Pre-computed solver baselines (without guidelines) are in `experiments/precompute_baselines.py` and the baseline files listed below.

## Project Structure

```
query-conditioned-guidelines/
├── experiments/
│   ├── prompts.py                  # Planner & solver system prompts (single source of truth)
│   ├── run_train.sh                # Main training entrypoint (calls verl.trainer.main_ppo)
│   ├── run_eval_math500.py         # MATH-500 evaluation (planner+solver or solver-only)
│   ├── prepare_numinamath.py       # Prepare NuminaMath-CoT 30k dataset
│   ├── prepare_dapomath.py         # Prepare DAPO-Math 30k dataset
│   ├── prepare_math500_eval.py     # Prepare MATH-500 eval set
│   ├── precompute_baselines.py     # Pre-compute solver baselines (no guideline)
│   ├── data/                       # Prepared datasets (parquet files)
│   │   ├── dapomath_30k/           # DAPO-Math 30k (train + test)
│   │   ├── dapomath_3k_prm/        # DAPO-Math 3k subset
│   │   ├── numinamath_30k/         # NuminaMath-CoT 30k
│   │   ├── numinamath_30k_v7/      # NuminaMath-CoT 30k (filtered v7)
│   │   └── math500_eval/           # MATH-500 eval set
│   └── logs/                       # Training & solver debug logs
├── verl/
│   └── utils/reward_score/
│       ├── guidelines.py           # Reward function: solver inference + accuracy scoring
│       └── math_verify.py          # math-verify wrapper for equivalence checking
├── slurm/                          # SLURM job scripts for DeltaAI cluster
├── containers/
│   ├── verl-vllm-24.04.sif         # Apptainer image (~12 GB) with verl + vLLM
│   └── *.def / Dockerfile*         # Container build recipes
├── config/
│   └── paths.sh                    # Local path config (verl, vllm overrides)
├── checkpoints/                    # Saved training checkpoints
└── wandb/                          # W&B run logs
```

## Datasets

| Dataset | Source | Size | Location |
|---------|--------|------|----------|
| DAPO-Math 30k | `BytedTsinghua-SIA/DAPO-Math-17k` (upsampled) | 30k train / 500 test | `experiments/data/dapomath_30k/` |
| DAPO-Math 3k | Subset of above | 3k train / 500 test | `experiments/data/dapomath_3k_prm/` |
| NuminaMath-CoT 30k | `AI-MO/NuminaMath-CoT` | 30k train / 500 test | `experiments/data/numinamath_30k/` |
| MATH-500 (eval) | `HuggingFaceH4/MATH-500` | 500 test | `experiments/data/math500_eval/` |

Prepare datasets:
```bash
python -m experiments.prepare_dapomath --output_dir experiments/data/dapomath_30k
python -m experiments.prepare_numinamath --output_dir experiments/data/numinamath_30k
python -m experiments.prepare_math500_eval
```

Each writes `train.parquet` and `test.parquet` with columns: `data_source`, `prompt` (list of chat messages), `reward_model` (dict with `style`, `ground_truth`, `extra_info`).

### Solver Baselines (no guideline)

Pre-computed solver accuracy on each question **without** any guideline, used for counterfactual reward experiments. Generated via `experiments/precompute_baselines.py`.

| File | Dataset | Entries |
|------|---------|---------|
| `experiments/data/numinamath_30k/baselines.json` | NuminaMath-CoT 30k | 30,000 |
| `experiments/data/dapomath_30k/baselines.json` | DAPO-Math 30k | 30,000 |
| `experiments/data/dapomath_3k_prm/prm_baselines.json` | DAPO-Math 3k (PRM scores) | 3,000 |

## Training

### Quick Start (interactive)

```bash
export DATA_DIR=experiments/data/dapomath_30k
export MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
export VERL_TOTAL_STEPS=50
export VERL_ROLLOUT_N=16
bash experiments/run_train.sh
```

### Via SLURM (DeltaAI)

```bash
sbatch --reservation=update slurm/run_train_50step_prm_v10.slurm
```

See `slurm/` for various job configurations. The general pattern:

1. SLURM script sets up container binds and environment variables
2. Calls `experiments/run_train.sh` with overrides
3. `run_train.sh` invokes `python3 -m verl.trainer.main_ppo` with GRPO config

### Training Configuration

Key environment variables consumed by `run_train.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `experiments/data/numinamath_30k` | Directory with `train.parquet` / `test.parquet` |
| `MODEL_PATH` | `Qwen/Qwen2.5-7B-Instruct` | Planner model (HuggingFace path or local) |
| `VERL_TOTAL_STEPS` | `300` | Total training steps |
| `VERL_TRAIN_BATCH_SIZE` | `60` | Questions per training step |
| `VERL_ROLLOUT_N` | `4` | Rollouts (guidelines) per question |
| `VERL_LR` | `1e-6` | Learning rate |
| `VERL_SAVE_FREQ` | `25` | Checkpoint save frequency (steps) |
| `VERL_TEST_FREQ` | `50` | Validation frequency (steps) |
| `VERL_CHECKPOINT_DIR` | `checkpoints/...` | Checkpoint output directory |
| `VERL_EXPERIMENT_SUFFIX` | auto | Suffix for wandb experiment name |
| `VERL_VAL_FILE` | `$DATA_DIR/test.parquet` | Validation file override |
| `WANDB_API_KEY` | (none) | Set to enable wandb logging |

### Reward Function Variables

Variables consumed by `verl/utils/reward_score/guidelines.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `VERL_GUIDELINES_SOLVER_MODEL` | `Qwen/Qwen2.5-14B-Instruct-AWQ` | Frozen solver model |
| `VERL_GUIDELINES_SOLVER_GPU_MEM` | `0.9` | GPU memory fraction for solver |
| `VERL_GUIDELINES_SOLVER_BATCH_SIZE` | (unlimited) | Cap on solver batch size |
| `VERL_GUIDELINES_USE_MATH_VERIFY` | `1` | Use math-verify for equivalence checking |
| `VERL_GUIDELINES_LOG_DIR` | (none) | If set, logs all solver outputs to JSONL |

### GPU Layout (4x GH200)

- **GPUs 0-2**: Training pool (actor + rollout + reference via FSDP)
- **GPU 3**: Frozen solver (vLLM offline inference, `gpu_memory_utilization=0.35`)

Set via `trainer.n_gpus_per_node=3` in `run_train.sh` (verl reserves 3 GPUs for training; the solver loads on the remaining GPU).

### Sampling

- **Planner rollouts**: `temperature=1.0`, `top_p=1.0`, `top_k=-1` (stochastic, for diversity across rollouts)
- **Solver inference**: `temperature=0.0` (greedy, deterministic)
- **Sequence lengths**: 3072 tokens prompt + 3072 tokens response (both planner and solver)

## Evaluation

Evaluate on MATH-500 with planner+solver or solver-only baseline:

```bash
# Planner + Solver
python -m experiments.run_eval_math500 \
  --model Qwen/Qwen2.5-3B-Instruct \
  --out_dir experiments/outputs/math500

# Solver-only baseline (no guideline)
python -m experiments.run_eval_math500 \
  --model Qwen/Qwen2.5-3B-Instruct \
  --no_guideline \
  --out_dir experiments/outputs/math500_baseline
```

Via SLURM:
```bash
sbatch slurm/eval_math500_planner_solver.slurm
sbatch slurm/eval_math500_solver_only.slurm
```

Outputs: `math500_guidelines.jsonl`, `math500_predictions.jsonl`, `math500_summary.txt`.

## Prompts

Defined in `experiments/prompts.py` (single source of truth for training and eval):

**Planner** (`PLANNER_SYSTEM_PROMPT`): Generates plain-English strategy only. No equations, no LaTeX, no numbers, no `\boxed{}`. Under 150 words.

**Solver** (`SOLVER_SYSTEM_PROMPT`): Follows the guideline, reasons inside `<think>` tags, puts final answer in `\boxed{}`.

## Container (DeltaAI / Apptainer)

The Apptainer image lives at:
```
containers/verl-vllm-24.04.sif   (~12 GB)
```

Build recipes in `containers/` (`Dockerfile.verl-vllm`, `verl-vllm.def`). The image includes verl, vLLM, PyTorch, and CUDA. See `containers/README-verl-vllm-build.md` for build instructions.

### Running in container

```bash
apptainer exec --nv \
  --bind /projects/bfgx/jparekh:/workspace \
  --bind /work/nvme/bfgx/$USER:/nvme \
  containers/verl-vllm-24.04.sif bash
```

Inside the container:
```bash
source /workspace/setup_verl_apptainer.sh
cd /workspace/query-conditioned-guidelines
pip install -q math-verify
export PYTHONPATH="/workspace/query-conditioned-guidelines:${PYTHONPATH}"
```

## DeltaAI Cluster Notes

- **GPUs**: NVIDIA GH200 120GB (ARM/aarch64)
- **Reservation**: Jobs require `--reservation=update` (or current active reservation)
- **NCCL**: Must use `NCCL_NET=Socket` (InfiniBand plugin crashes on GH200)
- **CUDA compat**: Container's CUDA compat lib must be hidden so host driver is used. SLURM scripts bind an empty dir over `/usr/local/cuda/compat/lib`.
- **NVMe**: Fast local storage at `/work/nvme/bfgx/$USER` -- used for checkpoints during training, then copied back to `/projects/` when jobs finish.
- **HF cache**: Set to `/tmp/hf_cache` inside container (node-local tmpfs).

## Reward Function Details

`verl/utils/reward_score/guidelines.py`:

1. **`strip_think_tags(text)`** -- Removes `<think>...</think>` blocks and `\boxed{}` from planner output to prevent answer leakage.
2. **`acc_reward(solver_output, ground_truth)`** -- Scores via math-verify (default) with mathruler fallback.
3. **`call_solver(guideline, question)`** -- Single-sample solver call (used by `compute_score`).
4. **`_run_solver_batch(...)`** -- Batch solver inference + scoring. Logs all samples to JSONL if `VERL_GUIDELINES_LOG_DIR` is set.
5. **`compute_score(predict_str, ground_truth, extra_info)`** -- Single-sample interface for verl.
6. **`compute_rewards_batch(guidelines, questions, ground_truths)`** -- Batch interface for verl. Supports chunked processing via `VERL_GUIDELINES_SOLVER_BATCH_SIZE`.

The solver vLLM instance is loaded once globally (`init_offline_llm()`) and reused across all reward calls.
