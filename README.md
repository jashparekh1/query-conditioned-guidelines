# Query-Conditioned Guidelines for Reasoning

This repository implements a two-stage approach for improving reasoning in large language models through query-conditioned guidelines. The method uses a **Planner** model to generate structured guidelines for solving problems, and a **Solver** model that follows these guidelines to produce answers.

## Overview

### Two-Stage Architecture

1. **Stage 1 - Guilder**: Generates structured, step-by-step guidelines for solving a given problem
2. **Stage 2 - Solver**: Follows the guidelines to reason step-by-step and produce the final answer

This approach enables better reasoning by decomposing problem-solving into explicit planning and execution phases.

## Getting Started

### Prerequisites

- Access to Delta cluster
- Python 3.10+
- CUDA-capable GPUs
- Singularity container with PyTorch 2.5.0.1

### Logging into Delta

1. **SSH into Delta**:
   ```bash
   ssh <your-username>@login1.delta.ncsa.illinois.edu
   ```

2. **Navigate to project directory**:
   ```bash
   cd /projects/bfgx/jparekh/query-conditioned-guidelines
   ```

3. **Load required modules** (if needed):
   ```bash
   module load singularity
   ```

### Environment Setup

1. **Activate the Singularity container**:
   ```bash
   CONTAINER=/projects/bfgx/jparekh/test_ngc/torch2501.sif
   singularity exec --nv \
     --bind /projects/bfgx/jparekh:/projects/bfgx/jparekh \
     --bind /dev/shm \
     $CONTAINER bash
   ```

2. **Activate the Python environment**:
   ```bash
   source verl_env/bin/activate
   ```

3. **Set environment variables**:
   ```bash
   export HF_HOME=/tmp/jparekh
   export TRANSFORMERS_CACHE=/tmp/jparekh
   export HF_DATASETS_CACHE=/tmp/jparekh
   export VLLM_WORKER_MULTIPROC_METHOD=spawn
   ```

4. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Update the paths in `config/paths.sh` according to your setup:

```bash
export PROJECT_ROOT=/projects/bfgx/jparekh/query-conditioned-guidelines
export VERL_PATH=/path/to/verl
export VLLM_PATH=/path/to/vllm
```

## Usage

### Training

#### Training with Guidelines (Two-Stage)

Train a model to generate guidelines using GRPO (Group Relative Policy Optimization):

```bash
bash shell_scripts/run_gsm8k_guideline.sh
```

This script:
- Trains a model on GSM8K dataset with guideline generation
- Uses Qwen2.5-1.5B-Instruct as the base model
- Saves checkpoints every 200 steps
- Logs to Weights & Biases

#### Training Baseline (Single-Stage)

Train a baseline model without guidelines:

```bash
bash shell_scripts/run_gsm8k.sh
```

### Evaluation

#### Two-Stage Evaluation

Evaluate the two-stage approach (Guilder + Solver):

**Option 1: Using SLURM (Recommended)**

```bash
sbatch slurm/eval_gsm8k.slurm
```

**Option 2: Interactive**

```bash
bash shell_scripts/eval.sh
```

**Option 3: Direct Python**

```bash
python eval.py \
  --guilder_model merge_models/offline-vllm-1.5b-step-498 \
  --solver_model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset gsm8k \
  --split test \
  --out_dir outputs/gsm8k_two_stage_outputs \
  --max_samples 500
```

#### Evaluation on Other Datasets

- **CommonSenseQA**: `sbatch slurm/eval_commonsenseqa.slurm`
- **StrategyQA**: `sbatch slurm/eval_strategyqa.slurm`

#### Evaluation Scripts

- `eval.py`: Main two-stage evaluation script
- `eval/gsm8k_guideline_eval.py`: Guideline-specific evaluation
- `eval/gsm8k_base_eval.py`: Baseline evaluation

### Evaluation Arguments

```bash
python eval.py \
  --guilder_model <path-to-guilder-model> \
  --solver_model <path-to-solver-model> \
  --dataset <gsm8k|commonsenseqa|strategyqa> \
  --split <train|test|validation> \
  --out_dir <output-directory> \
  --max_samples <number-of-samples> \
  --tensor_parallel_size <1-8> \
  --gpu_memory_utilization <0.0-1.0>
```

## Project Structure

```
query-conditioned-guidelines/
├── eval.py                          # Main two-stage evaluation script
├── eval/                            # Evaluation utilities
│   ├── gsm8k_base_eval.py          # Baseline evaluation
│   └── gsm8k_guideline_eval.py     # Guideline evaluation
├── outputs/                         # All evaluation results and outputs
│   ├── gsm8k_two_stage_outputs/    # GSM8K evaluation results
│   ├── math_two_stage_outputs/     # MATH evaluation results
│   ├── commonsenseqa_two_stage_outputs/  # CommonSenseQA results
│   ├── strategyqa_two_stage_outputs/     # StrategyQA results
│   └── *_eval_results.jsonl        # Evaluation result files
├── shell_scripts/                   # Shell scripts for training/evaluation
│   ├── run_gsm8k.sh                # Baseline training script
│   ├── run_gsm8k_guideline.sh      # Guideline training script
│   ├── run_eval_gsm8k.sh           # Evaluation script
│   ├── eval.sh                     # Interactive evaluation
│   └── serve_*.sh                   # Model serving scripts
├── slurm/                           # SLURM job scripts
│   ├── eval_gsm8k.slurm            # GSM8K evaluation job
│   ├── eval_math.slurm             # MATH evaluation job
│   ├── eval_commonsenseqa.slurm    # CommonSenseQA evaluation job
│   └── eval_strategyqa.slurm       # StrategyQA evaluation job
├── data_preprocessing/               # Data preprocessing scripts
│   ├── download_*.py               # Dataset download scripts
│   ├── convert_*.py                # Data conversion scripts
│   └── preprocess_*.sh              # Preprocessing scripts
├── config/                          # Configuration files
│   └── paths.sh                     # Path configuration
├── setup/                           # Setup and installation scripts
│   └── install_flash_attn.sh       # Flash attention installation
├── verl/                            # verl framework (RL training)
│   └── utils/reward_score/
│       └── guidelines.py            # Reward function for guidelines
└── requirements.txt                 # Python dependencies
```

## Key Components

### Guilder Model

The Guilder generates structured guidelines for problem-solving:

```
System Prompt: "You are a reasoning planner that creates structured, 
step-by-step guidelines for solving a given problem..."
```

### Solver Model

The Solver follows guidelines to produce answers:

```
System Prompt: "You are a careful and disciplined problem solver that 
follows a given guideline to reason step-by-step..."
```

### Reward Function

The reward function (`verl/utils/reward_score/guidelines.py`) evaluates:
- **Accuracy**: Exact match with ground truth
- **Format Compliance**: Proper use of `<think>` tags and `\boxed{}`
- **Brevity**: Length penalty (disabled by default)

## Datasets

Supported datasets:
- **GSM8K**: Grade school math problems
- **MATH**: High school and competition math problems
- **CommonSenseQA**: Commonsense reasoning questions
- **StrategyQA**: Strategic reasoning questions

## Training Details

- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Base Model**: Qwen2.5-1.5B-Instruct
- **Training Framework**: verl (Volcano Engine Reinforcement Learning)
- **Inference Engine**: vLLM (for efficient batch inference)

## Output Format

### Guidelines (JSONL)

```json
{
  "index": 0,
  "question": "What is 2+2?",
  "guideline": "1. Identify the operation\n2. Perform addition\n3. State the result"
}
```

### Predictions (JSONL)

```json
{
  "index": 0,
  "question": "What is 2+2?",
  "guideline": "...",
  "prediction": "<think>2+2=4</think>\n\\boxed{4}",
  "gold_answer": "4",
  "correct": true
}
```

## Troubleshooting

### CUDA Out of Memory

- Reduce `tensor_parallel_size`
- Lower `gpu_memory_utilization`
- Use smaller batch sizes

### Model Loading Issues

- Ensure models are in the correct format (HuggingFace)
- Check model paths in configuration
- Verify tokenizer compatibility

### SLURM Job Issues

- Check job logs in `logs/` directory
- Verify GPU allocation: `squeue -u $USER`
- Check resource limits: `sacct -j <job_id>`

## License

See `LICENSE` file for details.

## Acknowledgments

This project is built on top of:
- [verl](https://github.com/volcengine/verl): Reinforcement learning framework
- [vLLM](https://github.com/vllm-project/vllm): Efficient LLM inference
- [Qwen](https://github.com/QwenLM/Qwen): Base language models
