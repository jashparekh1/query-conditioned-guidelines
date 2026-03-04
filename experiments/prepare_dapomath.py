#!/usr/bin/env python3
"""
Prepare DAPO-Math-17k for guideline (planner) training.
- Loads BytedTsinghua-SIA/DAPO-Math-17k from HuggingFace
- Strips DAPO prompt boilerplate to extract clean question text
- Writes parquet in the same format as prepare_numinamath.py
  (data_source=guidelines, planner system prompt baked into user message)
- Uses a 500-question holdout as val split (or pass MATH500 eval via VERL_VAL_FILE)

Usage:
    python -m experiments.prepare_dapomath [--output_dir experiments/data/dapomath_17k] [--seed 42]
"""

import argparse
import os
import re

import datasets

from experiments.prompts import GUILDER_SYSTEM_PROMPT

DATASET_NAME = "BytedTsinghua-SIA/DAPO-Math-17k"
DATA_SOURCE = "guidelines"
SUBSAMPLE_SIZE = 30_000
VAL_SIZE = 500


def extract_question(raw_content: str) -> str:
    """Strip DAPO boilerplate from prompt content, returning just the problem text."""
    content = raw_content.strip()
    # Strip prefix: "Solve the following math problem ... where $Answer is the answer to the problem.\n\n"
    content = re.sub(
        r"^Solve the following math problem step by step\..*?where \$Answer is the answer to the problem\.\s*",
        "",
        content,
        flags=re.DOTALL,
    )
    # Strip suffix: "\n\nRemember to put your answer on its own line after 'Answer'."
    content = re.sub(
        r"\s*Remember to put your answer on its own line after [\"']?Answer:[\"']?.*$",
        "",
        content,
        flags=re.DOTALL,
    )
    return content.strip()


def make_row(problem: str, ground_truth: str, idx: int, split: str) -> dict:
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
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare DAPO-Math-17k for guideline training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/data/dapomath_30k",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=SUBSAMPLE_SIZE,
        help=f"Number of train examples to sample (default {SUBSAMPLE_SIZE})",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    print(f"Loading {DATASET_NAME} ...")
    ds = datasets.load_dataset(
        DATASET_NAME,
        split="train",
        cache_dir="/tmp/hf_cache",
    )
    print(f"Raw dataset size: {len(ds):,}")

    # Subsample indices up front to avoid iterating all 1.7M rows
    n_sample = min(args.subsample + VAL_SIZE + 5000, len(ds))  # oversample slightly for filtering losses
    sampled_indices = random.sample(range(len(ds)), n_sample)
    print(f"Sampling {n_sample:,} rows to extract {args.subsample + VAL_SIZE} valid problems ...")

    problems = []
    ground_truths = []
    skipped = 0
    for i in sampled_indices:
        row = ds[i]
        raw = row["prompt"][0]["content"] if isinstance(row["prompt"], list) else row["prompt"]["content"]
        q = extract_question(raw)
        gt = row["reward_model"]["ground_truth"]
        if not q or not gt:
            skipped += 1
            continue
        problems.append(q)
        ground_truths.append(str(gt))
        if len(problems) >= args.subsample + VAL_SIZE:
            break

    print(f"Valid problems collected: {len(problems)} (skipped {skipped} empty)")

    # Shuffle and split
    indices = list(range(len(problems)))
    random.shuffle(indices)

    val_idx = indices[:VAL_SIZE]
    train_idx = indices[VAL_SIZE:VAL_SIZE + args.subsample]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    print(f"Sample GTs (first 10): {[ground_truths[i] for i in train_idx[:10]]}")
    print(f"Sample Q: {problems[train_idx[0]][:150]}")

    train_rows = [make_row(problems[i], ground_truths[i], j, "train") for j, i in enumerate(train_idx)]
    val_rows = [make_row(problems[i], ground_truths[i], j, "test") for j, i in enumerate(val_idx)]

    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = datasets.Dataset.from_list(train_rows)
    train_path = os.path.join(args.output_dir, "train.parquet")
    train_ds.to_parquet(train_path)
    print(f"Wrote {train_path} ({len(train_ds)} rows)")

    val_ds = datasets.Dataset.from_list(val_rows)
    val_path = os.path.join(args.output_dir, "test.parquet")
    val_ds.to_parquet(val_path)
    print(f"Wrote {val_path} ({len(val_ds)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
