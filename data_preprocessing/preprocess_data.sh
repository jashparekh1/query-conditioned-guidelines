#!/bin/bash
# Preprocess GSM8K data for guidelines training
# This will create train.parquet and test.parquet with correct format

set -e

CONTAINER=/projects/bfgx/jparekh/test_ngc/torch2501.sif
PROJECT_ROOT=/projects/bfgx/jparekh/query-conditioned-guidelines
DATA_DIR=$PROJECT_ROOT/data/gsm8k

echo "========================================="
echo "Preprocessing GSM8K data for guidelines"
echo "========================================="

cd /projects/bfgx/jparekh/test_ngc

# Run conversion script (it will handle backups itself)
echo "Converting existing parquet files to guidelines format..."
singularity exec --nv \
  --bind /projects/bfgx/jparekh:/projects/bfgx/jparekh \
  $CONTAINER bash -c "
    source verl_env/bin/activate
    cd $PROJECT_ROOT
    python3 convert_existing_data.py
  "

echo "========================================="
echo "Preprocessing complete!"
echo "New files created in: $DATA_DIR"
echo "========================================="

