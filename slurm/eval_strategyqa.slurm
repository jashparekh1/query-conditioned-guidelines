#!/bin/bash
#SBATCH --account=bfgx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=eval_sqa
#SBATCH --output=logs/eval_strategyqa_%j.out
#SBATCH --error=logs/eval_strategyqa_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Create logs directory
mkdir -p logs

# Set HuggingFace cache to /tmp/jparekh
mkdir -p /tmp/jparekh
export HF_HOME=/tmp/jparekh
export TRANSFORMERS_CACHE=/tmp/jparekh
export HF_DATASETS_CACHE=/tmp/jparekh

# Use 1 GPU (single GPU to avoid memory corruption issues)
export CUDA_VISIBLE_DEVICES=0
TENSOR_PARALLEL_SIZE=1

# Paths
CONTAINER=/projects/bfgx/jparekh/test_ngc/torch2501.sif
PROJECT_ROOT=/projects/bfgx/jparekh/query-conditioned-guidelines

echo "=========================================="
echo "StrategyQA Evaluation (500 problems)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Time limit: 12 hours"
echo "=========================================="

cd /projects/bfgx/jparekh/test_ngc

singularity exec --nv \
  --bind /projects/bfgx/jparekh:/projects/bfgx/jparekh \
  --bind /dev/shm \
  --env HF_HOME=/tmp/jparekh \
  --env TRANSFORMERS_CACHE=/tmp/jparekh \
  --env HF_DATASETS_CACHE=/tmp/jparekh \
  --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  --env PYTHONUNBUFFERED=1 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  $CONTAINER bash -c "
    source verl_env/bin/activate
    cd $PROJECT_ROOT
    python eval.py \
      --guilder_model merge_models/offline-vllm-1.5b-step-498 \
      --solver_model Qwen/Qwen2.5-1.5B-Instruct \
      --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
      --split train \
      --out_dir outputs/strategyqa_two_stage_outputs \
      --dataset strategyqa \
      --max_samples 500
  "

echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: outputs/strategyqa_two_stage_outputs/"
echo "=========================================="

