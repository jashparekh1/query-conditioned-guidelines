#!/usr/bin/env python3
"""
Prepare a 500-question eval set from qwedsacf/competition_math (MATH dataset).
Applies the same is_gradable filter as prepare_numinamath.py to keep only
clean numeric/fraction/expression answers that math_verify can handle.

Output: experiments/data/math500_eval/test.parquet
"""

import argparse
import os
import re

import datasets

from experiments.prompts import GUILDER_SYSTEM_PROMPT
from experiments.prepare_numinamath import extract_boxed, is_gradable

DATASET_NAME = "qwedsacf/competition_math"
EVAL_SIZE = 500
DATA_SOURCE = "guidelines"


def main():
    parser = argparse.ArgumentParser(description="Prepare MATH eval set for guideline evaluation")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/data/math500_eval",
        help="Directory to write test.parquet",
    )
    parser.add_argument("--n", type=int, default=EVAL_SIZE, help=f"Number of eval examples (default {EVAL_SIZE})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    args = parser.parse_args()

    print(f"Loading {DATASET_NAME}...")
    # Cache to /tmp to avoid home dir quota issues
    ds = datasets.load_dataset(DATASET_NAME, cache_dir="/tmp/hf_cache")
    all_data = ds["train"]  # only split available
    print(f"Total problems: {len(all_data)}")

    # Filter to gradable problems
    kept_indices = []
    rejected = {"empty_gt": 0, "no_boxed": 0, "not_gradable": 0}
    for i in range(len(all_data)):
        example = all_data[i]
        problem = example.get("problem", "")
        solution = example.get("solution", "")
        gt = extract_boxed(solution)

        if not gt:
            rejected["no_boxed"] += 1
            continue

        gt = gt.strip()
        if not gt:
            rejected["empty_gt"] += 1
            continue

        if not is_gradable(gt, problem):
            rejected["not_gradable"] += 1
            continue

        kept_indices.append(i)

    print(f"Filtered: {len(all_data)} -> {len(kept_indices)} ({len(all_data) - len(kept_indices)} removed)")
    print(f"  Rejection reasons: {rejected}")

    # Subsample
    filtered = all_data.select(kept_indices)
    if len(filtered) > args.n:
        filtered = filtered.shuffle(seed=args.seed).select(range(args.n))
    print(f"Using {len(filtered)} eval examples")

    # Build rows in the same format as training data
    def make_row(example: dict, idx: int) -> dict:
        problem = example["problem"]
        solution = example["solution"]
        ground_truth = extract_boxed(solution).strip()
        user_content = GUILDER_SYSTEM_PROMPT + "\n\nQuestion: " + problem
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": user_content}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": "test",
                "index": idx,
                "question": problem,
                "solution": solution,
                "level": example.get("level", ""),
                "type": example.get("type", ""),
            },
        }

    eval_dataset = [make_row(filtered[i], i) for i in range(len(filtered))]
    eval_df = datasets.Dataset.from_list(eval_dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "test.parquet")
    eval_df.to_parquet(out_path)
    print(f"Wrote {out_path} ({len(eval_df)} rows)")

    # Print samples
    print("\nSample ground truths:")
    for i in range(min(10, len(eval_dataset))):
        gt = eval_dataset[i]["reward_model"]["ground_truth"]
        q = eval_dataset[i]["extra_info"]["question"][:80]
        lvl = eval_dataset[i]["extra_info"]["level"]
        typ = eval_dataset[i]["extra_info"]["type"]
        print(f"  [{i}] GT: {gt:>30s}  ({lvl}, {typ})  Q: {q}...")

    # Stats by level and type
    levels = {}
    types = {}
    for row in eval_dataset:
        lvl = row["extra_info"]["level"]
        typ = row["extra_info"]["type"]
        levels[lvl] = levels.get(lvl, 0) + 1
        types[typ] = types.get(typ, 0) + 1
    print(f"\nBy level: {dict(sorted(levels.items()))}")
    print(f"By type: {dict(sorted(types.items()))}")
    print("Done.")


if __name__ == "__main__":
    main()
