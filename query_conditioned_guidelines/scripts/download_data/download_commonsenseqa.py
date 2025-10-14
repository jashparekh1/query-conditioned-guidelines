#!/usr/bin/env python3
"""Download CommonsenseQA dataset"""
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Setup paths
data_dir = Path(__file__).parent.parent / "data" / "commonsenseqa"
data_dir.mkdir(parents=True, exist_ok=True)

print("Downloading CommonsenseQA dataset...")
dataset = load_dataset("tau/commonsense_qa")

print("Converting train split...")
train_df = pd.DataFrame(dataset['train'])
train_df.to_parquet(data_dir / "train.parquet", index=False)

print("Converting validation split (using as test)...")
test_df = pd.DataFrame(dataset['test'])
test_df.to_parquet(data_dir / "test.parquet", index=False)

print(f"✓ Saved to {data_dir}")
print(f"  Train: {len(train_df)} samples")
print(f"  Test: {len(test_df)} samples")