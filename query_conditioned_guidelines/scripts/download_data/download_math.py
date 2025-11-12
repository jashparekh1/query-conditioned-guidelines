#!/usr/bin/env python3
"""Download MATH dataset"""
from datasets import load_dataset
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Setup paths
data_dir = Path(__file__).parent.parent.parent / "data" / "math"
data_dir.mkdir(parents=True, exist_ok=True)

print("Downloading MATH dataset...")
dataset = load_dataset("qwedsacf/competition_math")

print("Splitting dataset (80/20 train/test)...")
df = pd.DataFrame(dataset['train'])
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print("Saving splits...")
train_df.to_parquet(data_dir / "train.parquet", index=False)
test_df.to_parquet(data_dir / "test.parquet", index=False)

print(f"✓ Saved to {data_dir}")
print(f"  Train: {len(train_df)} samples")
print(f"  Test: {len(test_df)} samples")