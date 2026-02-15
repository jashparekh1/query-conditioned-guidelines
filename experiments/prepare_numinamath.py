#!/usr/bin/env python3
"""
Prepare NuminaMath-CoT for guideline (guilder) training.
- Loads AI-MO/NuminaMath-CoT from HuggingFace
- Subsamples 30k from train
- Extracts ground truth from solution (\\boxed{} or last line)
- Writes parquet in the same format as gsm8k_guidelines_processed (data_source=guidelines)
"""

import argparse
import os
import re

import datasets

# Use shared prompts so training and eval match
from experiments.prompts import GUILDER_SYSTEM_PROMPT

DATASET_NAME = "AI-MO/NuminaMath-CoT"
SUBSAMPLE_SIZE = 30_000
DATA_SOURCE = "guidelines"


def extract_boxed(s: str) -> str | None:
    """Extract content of last \\boxed{...} in solution string."""
    if not s or not s.strip():
        return None
    # Match \boxed{...} - may contain nested braces
    matches = list(re.finditer(r"\\boxed\{", s))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(s) and depth > 0:
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return s[start : i - 1].strip()


def extract_ground_truth(solution_str: str) -> str:
    """
    Extract ground truth answer from NuminaMath solution.
    Prefer \\boxed{...}; otherwise use last non-empty line (some sources format differently).
    """
    boxed = extract_boxed(solution_str)
    if boxed:
        return boxed
    lines = [ln.strip() for ln in solution_str.strip().split("\n") if ln.strip()]
    if lines:
        last = lines[-1]
        # Remove common wrappers
        if last.startswith("$") and last.endswith("$"):
            last = last[1:-1]
        return last
    return ""


def main():
    parser = argparse.ArgumentParser(description="Prepare NuminaMath-CoT for guideline training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/data/numinamath_30k",
        help="Directory to write train.parquet and test.parquet",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=SUBSAMPLE_SIZE,
        help=f"Number of train examples to keep (default {SUBSAMPLE_SIZE})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    args = parser.parse_args()

    print(f"Loading {DATASET_NAME}...")
    ds = datasets.load_dataset(DATASET_NAME, "default", trust_remote_code=True)
    train_raw = ds["train"]
    test_raw = ds["test"] if "test" in ds and len(ds["test"]) > 0 else None

    n_total = len(train_raw)
    if args.subsample < n_total:
        train_raw = train_raw.shuffle(seed=args.seed).select(range(args.subsample))
        print(f"Subsampled to {len(train_raw)} train examples")
    else:
        print(f"Using full train set ({len(train_raw)} examples)")

    def make_row(example: dict, idx: int, split: str) -> dict:
        problem = example.get("problem", "")
        solution = example.get("solution", "")
        ground_truth = extract_ground_truth(solution)
        user_content = GUILDER_SYSTEM_PROMPT + "\n\nQuestion: " + problem
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": user_content}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
                "question": problem,
                "solution": solution,
            },
        }

    train_dataset = [make_row(train_raw[i], i, "train") for i in range(len(train_raw))]
    train_df = datasets.Dataset.from_list(train_dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    train_df.to_parquet(train_path)
    print(f"Wrote {train_path} ({len(train_df)} rows)")

    if test_raw is not None and len(test_raw) > 0:
        test_dataset = [make_row(test_raw[i], i, "test") for i in range(len(test_raw))]
        test_df = datasets.Dataset.from_list(test_dataset)
        test_path = os.path.join(args.output_dir, "test.parquet")
        test_df.to_parquet(test_path)
        print(f"Wrote {test_path} ({len(test_df)} rows)")
    else:
        # No test split: use a small holdout from train for val
        n_val = min(500, len(train_df) // 10)
        val_df = train_df.select(range(n_val))
        val_path = os.path.join(args.output_dir, "test.parquet")
        val_df.to_parquet(val_path)
        print(f"No test split in dataset; wrote {val_path} as val ({len(val_df)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
