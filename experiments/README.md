# Experiments: NuminaMath-CoT training + MATH-500 evaluation

This folder contains the pipeline for:

1. **Training** the guilder (planner) on NuminaMath-CoT (30k subsample) for 300 steps with 3072-token sequences, using Qwen2.5-3B-Instruct as both planner and solver. Checkpoints are saved every 10 steps.
2. **Evaluating** on MATH-500 with the same prompts and MathVerify for grading.

Prompts are shared between training and evaluation via `experiments/prompts.py` so they stay identical.

## Setup

**Python >= 3.10** required. Load your Python module first on HPC (e.g. `module load python/3.11.9`).

### New venv (recommended)

From the repository root:

```bash
./scripts/setup_venv.sh
source venv/bin/activate
```

On a CUDA node, include CUDA extras:

```bash
./scripts/setup_venv.sh cuda
source venv/bin/activate
```

**vLLM (for MATH-500 eval):** The base requirements omit vLLM so the venv installs on ARM login nodes. vLLM has prebuilt wheels only for x86_64; on ARM it builds from source and can fail (e.g. compiler `-march` error). Options:

- On an **x86_64 GPU node** where wheels exist: `pip install -r requirements-vllm.txt`
- **Apptainer (recommended on HPC):** Use a container image that already has vLLM. See **[containers/README.md](../containers/README.md)** for building and running with Apptainer.

The script removes any existing `venv`, creates a fresh one, and installs `requirements.txt` (and optionally `requirements-cuda.txt`).

### Use existing venv

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## API keys

- **Weights & Biases (WANDB)**  
  Used when `trainer.logger=wandb`. Set `WANDB_API_KEY` so Ray workers can log:
  ```bash
  export WANDB_API_KEY=your_key_here
  ./experiments/run_train.sh
  ```
  To disable WANDB, override: `trainer.logger=console` (or edit `run_train.sh`).

- **Hugging Face (HF)**  
  Only needed for gated or private models. For public models (e.g. `Qwen/Qwen2.5-3B-Instruct`) no token is required. If you use a gated/private model, set one of:
  - `HF_TOKEN`, or  
  - `HUGGING_FACE_HUB_TOKEN`
  so the trainer and workers can download weights.

## 1. Prepare data

From the **repository root**:

```bash
export PYTHONPATH=.
python -m experiments.prepare_numinamath --output_dir experiments/data/numinamath_30k
```

This downloads AI-MO/NuminaMath-CoT, subsamples 30k train examples, and writes `train.parquet` and `test.parquet` in the guidelines format expected by the trainer.

## 2. Train

**Nodes/GPUs:** 1 node × 4 GPUs (same as previous GSM8k guideline runs). Total GPUs = `trainer.nnodes × trainer.n_gpus_per_node`.

Run from the **repository root** (e.g. `/projects/bfgx/jparekh/query-conditioned-guidelines`).

### 20-step smoke test (e.g. 2hr node)

```bash
export WANDB_API_KEY=your_key   # optional
./experiments/run_train_20step_test.sh
```

This runs 20 steps, saves checkpoints at steps 10 and 20, and logs to `experiments/logs/train_20step_test_*.log`. Checkpoints go to `checkpoints/query-conditioned-guidelines/numinamath30k-3b-20step-test/`.

### Full 300-step run

```bash
export WANDB_API_KEY=your_key   # optional
./experiments/run_train.sh
```

For the full job you can submit via sbatch instead of running interactively.

- **Model:** Qwen2.5-3B-Instruct (planner and solver; solver is used for reward via `VERL_GUIDELINES_SOLVER_MODEL`).
- **Steps:** 300.
- **Sequence length:** 3072 (prompt and response).
- **Checkpointing:** every 10 steps (saved under `checkpoints/query-conditioned-guidelines/numinamath30k-3b-300steps/`).
- Logs are written to `experiments/logs/train_*.log`.

Override paths or scale to more nodes:

```bash
DATA_DIR=/path/to/numinamath_30k MODEL_PATH=Qwen/Qwen2.5-3B-Instruct ./experiments/run_train.sh
# Multi-node (e.g. 2 nodes × 4 GPUs): ensure Ray cluster is running across nodes, then:
./experiments/run_train.sh trainer.nnodes=2 trainer.n_gpus_per_node=4
```

## 3. Evaluate on MATH-500 (MathVerify)

From the **repository root**:

```bash
PYTHONPATH=. python -m experiments.run_eval_math500 --model Qwen/Qwen2.5-3B-Instruct --max_samples 100
```

Options:

- `--model`: Model path (same for guilder and solver). Default: `Qwen/Qwen2.5-3B-Instruct`.
- `--max_samples`: Cap number of MATH-500 examples (default: use full test set).
- `--out_dir`: Output directory. Default: `experiments/outputs/math500`.
- `--tensor_parallel_size`, `--gpu_memory_utilization`, `--batch_size`, `--guilder_max_new_tokens`, `--solver_max_new_tokens`, etc.

Outputs:

- `{out_dir}/math500_guidelines.jsonl`: Guilder outputs.
- `{out_dir}/math500_predictions.jsonl`: Solver outputs + per-example MathVerify score.
- `{out_dir}/math500_summary.txt`: Number of examples and MathVerify accuracy.

**Requirement:** `pip install math-verify` for MathVerify grading.

## Prompts

- **Guilder:** `experiments/prompts.py` → `GUILDER_SYSTEM_PROMPT`
- **Solver:** `experiments/prompts.py` → `SOLVER_SYSTEM_PROMPT`

These are used by:

- `experiments/prepare_numinamath.py` (prompt text in parquet).
- `verl/utils/reward_score/guidelines.py` (solver prompt and model).
- `eval.py` (imports from `experiments.prompts` when available).
- `experiments/run_eval_math500.py`.

So training and evaluation use the same prompts.
