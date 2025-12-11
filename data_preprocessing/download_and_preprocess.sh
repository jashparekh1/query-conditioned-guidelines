#!/bin/bash
# Download GSM8K from HuggingFace and preprocess for guidelines training
# This will create train.parquet and test.parquet with correct format

set -e

CONTAINER=/projects/bfgx/jparekh/test_ngc/torch2501.sif
PROJECT_ROOT=/projects/bfgx/jparekh/query-conditioned-guidelines
DATA_DIR=$PROJECT_ROOT/data/gsm8k
HF_HOME=/projects/bfgx/jparekh/causal-inference-project/v2/cache

echo "========================================="
echo "Downloading and preprocessing GSM8K data"
echo "========================================="

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Backup old files if they exist
if [ -f "$DATA_DIR/train.parquet" ]; then
    echo "Backing up old train.parquet..."
    mv "$DATA_DIR/train.parquet" "$DATA_DIR/train.parquet.backup.$(date +%Y%m%d_%H%M%S)"
fi

if [ -f "$DATA_DIR/test.parquet" ]; then
    echo "Backing up old test.parquet..."
    mv "$DATA_DIR/test.parquet" "$DATA_DIR/test.parquet.backup.$(date +%Y%m%d_%H%M%S)"
fi

cd /projects/bfgx/jparekh/test_ngc

# Run direct download script - downloads in memory, no cache
echo "Downloading GSM8K from HuggingFace and preprocessing..."
echo "Downloading directly in memory (no cache files saved)..."
singularity exec --nv \
  --bind /projects/bfgx/jparekh:/projects/bfgx/jparekh \
  $CONTAINER bash -c "
    source verl_env/bin/activate
    cd $PROJECT_ROOT
    python3 download_direct.py
  "

echo "========================================="
echo "✓ Preprocessing complete!"
echo "Files created in: $DATA_DIR"
echo "========================================="

# Verify the files
echo ""
echo "Verifying converted files..."
singularity exec --nv \
  --bind /projects/bfgx/jparekh:/projects/bfgx/jparekh \
  $CONTAINER bash -c "
    source verl_env/bin/activate
    python3 -c \"
import datasets
train = datasets.load_dataset('parquet', data_files='$DATA_DIR/train.parquet')['train']
test = datasets.load_dataset('parquet', data_files='$DATA_DIR/test.parquet')['train']
print('Train examples:', len(train))
print('Test examples:', len(test))
print('Sample train keys:', list(train[0].keys()))
print('Sample train data_source:', train[0]['data_source'])
print('Sample train extra_info keys:', list(train[0]['extra_info'].keys()))
print('Sample train reward_model:', train[0]['reward_model'])
\"
  "

echo ""
echo "========================================="
echo "✓ All done! Data is ready for training."
echo "========================================="

