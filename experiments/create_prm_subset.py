#!/usr/bin/env python3
"""Create a training subset parquet containing only questions that have PRM baselines.

Usage:
    python -m experiments.create_prm_subset \
        --baselines experiments/data/dapomath_30k/prm_baselines.json \
        --input experiments/data/dapomath_30k/train.parquet \
        --output experiments/data/dapomath_3k/train.parquet
"""
import argparse
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", required=True, help="PRM baselines JSON")
    parser.add_argument("--input", required=True, help="Full train.parquet")
    parser.add_argument("--output", required=True, help="Output subset train.parquet")
    args = parser.parse_args()

    with open(args.baselines) as f:
        baselines = json.load(f)

    baseline_indices = sorted(int(k) for k in baselines.keys())
    print(f"PRM baselines cover {len(baseline_indices)} questions")

    df = pd.read_parquet(args.input)
    print(f"Full dataset: {len(df)} rows")

    subset = df.iloc[baseline_indices].reset_index(drop=True)
    print(f"Subset: {len(subset)} rows")

    # Rewrite extra_info indices to match new row positions
    for i in range(len(subset)):
        info = dict(subset.iloc[i]["extra_info"])
        info["original_index"] = baseline_indices[i]
        info["index"] = i
        subset.at[i, "extra_info"] = info

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    subset.to_parquet(args.output)
    print(f"Wrote {args.output}")

    # Also remap baselines to new indices
    new_baselines = {}
    for new_idx, old_idx in enumerate(baseline_indices):
        new_baselines[str(new_idx)] = baselines[str(old_idx)]
    new_baselines_path = os.path.join(os.path.dirname(args.output), "prm_baselines.json")
    with open(new_baselines_path, "w") as f:
        json.dump(new_baselines, f)
    print(f"Wrote remapped baselines: {new_baselines_path}")


if __name__ == "__main__":
    main()
