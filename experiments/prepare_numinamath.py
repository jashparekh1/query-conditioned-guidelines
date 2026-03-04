#!/usr/bin/env python3
"""
Prepare NuminaMath-CoT for guideline (guilder) training.
- Loads AI-MO/NuminaMath-CoT from HuggingFace
- Filters to problems with clean, gradable ground truths (numbers, fractions, expressions)
- Rejects proofs, text answers, multiple-choice letters, and empty/ungradable answers
- Subsamples 30k from the filtered set
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


def is_gradable(ground_truth: str, problem: str = "") -> bool:
    """
    Check if a ground truth answer can be reliably graded by math_verify.

    Accepts: numbers, fractions, algebraic expressions, coordinates, sets of numbers.
    Rejects: proofs, text answers, single letters (multiple choice), empty, long text.
    """
    gt = ground_truth.strip()

    # Empty or whitespace-only
    if not gt:
        return False

    # Too short (single letter = multiple choice like "A", "B", "C", "D")
    # Allow single digits and common math like "0", "1", "e", but reject plain letters
    if len(gt) == 1 and gt.isalpha() and gt not in ("e", "i"):
        return False

    # Multiple-choice pattern: "A", "B)", "C:", "(D)", etc.
    if re.match(r"^[\(\[]?[A-E][\)\]:\.]?$", gt):
        return False
    # "Option A", "Choice B", etc.
    if re.match(r"^(option|choice|answer)\s*[A-E]", gt, re.IGNORECASE):
        return False
    # MC letter followed by value: "C) 20", "A. 15", "B: 42", "(D) 100"
    if re.match(r"^[\(\[]?[A-E][\)\]:\.\s]+\s*\S", gt):
        return False
    # LaTeX-formatted multiple choice: \textbf{(C)} 26.67, \text{(A)}, etc.
    if re.search(r"\\text(?:bf|rm|it|sf)?\s*\{\s*\(?[A-E]\)?\s*\}", gt):
        return False

    # Ratio notation: "2:1", "3:4:5", etc. — math_verify can't parse these
    if re.match(r"^\d+\s*:\s*\d+(\s*:\s*\d+)*$", gt):
        return False

    # Reject \text{...} containing words (text answers like \text{peach}, \text{yes})
    text_contents = re.findall(r"\\text(?:bf|rm|it|sf)?\s*\{([^}]*)\}", gt)
    for tc in text_contents:
        # If the \text{} contains alphabetic words (not just math like \text{cm}),
        # and those words are real English words (>2 chars), reject
        words = re.findall(r"[a-zA-Z]{3,}", tc)
        # Allow common math/unit abbreviations
        math_abbrevs = {"mod", "gcd", "lcm", "max", "min", "log", "sin", "cos", "tan",
                        "exp", "det", "dim", "deg", "rem", "and"}
        non_math_words = [w for w in words if w.lower() not in math_abbrevs]
        if non_math_words:
            return False

    # Reject multi-valued answers: "3 or 7", "x = 1 \text{ or } x = 2"
    if re.search(r"\bor\b", gt, re.IGNORECASE):
        return False
    # Comma-separated multiple answers like "1, 2" or "a = 1, b = 2"
    # But allow commas inside LaTeX like \frac{1,2} or coordinates (1,2)
    # Heuristic: if there's a comma NOT inside braces/parens, it's multi-valued
    gt_no_braces = re.sub(r"\{[^}]*\}", "", gt)
    gt_no_braces = re.sub(r"\([^)]*\)", "", gt_no_braces)
    if "," in gt_no_braces:
        return False

    # Reject if ground truth is mostly text (proofs, explanations)
    # Heuristic: if >60% of non-whitespace chars are alphabetic (excluding LaTeX commands),
    # it's probably a text answer
    stripped_latex = re.sub(r"\\[a-zA-Z]+", "", gt)  # remove \frac, \text, \mathbf, etc.
    stripped_latex = re.sub(r"[{}()\\$\[\]|,;:=<>+\-*/^_\d.\s]", "", stripped_latex)
    if len(stripped_latex) > 20:
        # Lots of alphabetic content remaining = probably text/proof
        return False

    # Reject common proof/text indicators in the ground truth itself
    text_indicators = [
        "prove", "proof", "shown", "therefore", "hence", "thus",
        "is a convex", "is a polygon", "edges", "is true", "QED",
        "the answer is", "we conclude", "it follows",
    ]
    gt_lower = gt.lower()
    for indicator in text_indicators:
        if indicator in gt_lower:
            return False

    # Reject inequalities: ≥, ≤, >, <, \geq, \leq, \ge, \le in GT
    if re.search(r"\\geq|\\leq|\\ge(?!t)|\\le(?!t)|\\gt|\\lt|[≥≤><]", gt):
        return False

    # Reject currency symbols in GT — math_verify can't parse "$178.88"
    if re.search(r"\\?\$|\\text\{\\?\$\}", gt):
        return False

    # Reject set notation: \{...\}, \emptyset — math_verify can't grade these
    if re.search(r"\\\{|\\\}|\\emptyset", gt):
        return False

    # Reject percent signs — "30\%" won't match solver's "30" or "0.3"
    if re.search(r"\\%|%", gt):
        return False

    # Reject subscripted variables like "16a_1^2" — symbolic, fragile grading
    if re.search(r"[a-zA-Z]_\{?\d", gt):
        return False

    # Reject symbolic identities/equations: GT contains "=" (e.g. "n(n+2)+1=(n+1)^2")
    # These are proofs/derivations, not numeric answers. Allow "x = 5" style (value after =).
    if "=" in gt:
        # If the part after the last "=" is a simple number/expression, keep it
        # e.g. "x = 5" is fine, "n(n+2)+1 = (n+1)^2" is not
        after_eq = gt.rsplit("=", 1)[-1].strip()
        # If after_eq still contains variables (letters not in LaTeX commands), reject
        after_clean = re.sub(r"\\[a-zA-Z]+", "", after_eq)  # strip \frac etc
        after_clean = re.sub(r"[{}()\[\]\\$+\-*/^_\s\d.,]", "", after_clean)
        if len(after_clean) > 2:  # more than 2 letters remaining = symbolic
            return False

    # Reject if ground truth has no mathematical content at all
    # Must contain at least one digit, or a fraction/sqrt/pi/common math symbol
    has_math = bool(re.search(r"\d|\\frac|\\sqrt|\\pi|\\infty|\\pm", gt))
    if not has_math:
        # Allow simple symbolic answers like "x", "n+1", but not long text
        if len(gt) > 30:
            return False
        # Must look like a short expression
        if not re.match(r"^[a-z0-9\s\\{}()+\-*/^_=<>,.|]+$", gt, re.IGNORECASE):
            return False

    # Reject problem text that looks like a proof request or has multiple blanks
    if problem:
        prob_lower = problem.lower()
        proof_phrases = ["prove that", "show that", "prove:", "proof"]
        for phrase in proof_phrases:
            if phrase in prob_lower:
                return False
        # Multiple blanks = multiple answers expected, GT likely incomplete
        blank_count = problem.count("______") + problem.count("____")
        if blank_count > 1:
            return False
        # Multiple-choice problems: textbf{(A)}, (A), (B), etc. — GT may be unreliable
        if re.search(r"\\textbf\s*\{\s*\(?[A-E]\)?\s*\}", problem):
            return False

    return True


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

    # Filter to gradable problems
    n_before = len(train_raw)
    kept_indices = []
    rejected_reasons = {"empty_gt": 0, "no_boxed": 0, "not_gradable": 0, "proof": 0}
    for i in range(n_before):
        example = train_raw[i]
        problem = example.get("problem", "")
        solution = example.get("solution", "")
        gt = extract_ground_truth(solution)

        if not gt:
            rejected_reasons["empty_gt"] += 1
            continue

        # Prefer problems that have \boxed{} in the solution (cleaner ground truth)
        if not extract_boxed(solution):
            rejected_reasons["no_boxed"] += 1
            continue

        if not is_gradable(gt, problem):
            rejected_reasons["not_gradable"] += 1
            continue

        kept_indices.append(i)

    train_filtered = train_raw.select(kept_indices)
    n_after = len(train_filtered)
    print(f"Filtered: {n_before} -> {n_after} ({n_before - n_after} removed)")
    print(f"  Rejection reasons: {rejected_reasons}")

    if n_after < args.subsample:
        print(f"Warning: only {n_after} gradable examples available (requested {args.subsample})")
        train_sub = train_filtered.shuffle(seed=args.seed)
    else:
        train_sub = train_filtered.shuffle(seed=args.seed).select(range(args.subsample))
    print(f"Using {len(train_sub)} train examples")

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

    train_dataset = [make_row(train_sub[i], i, "train") for i in range(len(train_sub))]
    train_df = datasets.Dataset.from_list(train_dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    train_df.to_parquet(train_path)
    print(f"Wrote {train_path} ({len(train_df)} rows)")

    # For test set, apply same filtering
    if test_raw is not None and len(test_raw) > 0:
        test_kept = []
        for i in range(len(test_raw)):
            example = test_raw[i]
            problem = example.get("problem", "")
            solution = example.get("solution", "")
            gt = extract_ground_truth(solution)
            if gt and extract_boxed(solution) and is_gradable(gt, problem):
                test_kept.append(i)
        test_filtered = test_raw.select(test_kept)
        # Cap test at 500 for validation speed
        if len(test_filtered) > 500:
            test_filtered = test_filtered.shuffle(seed=args.seed).select(range(500))
        test_dataset = [make_row(test_filtered[i], i, "test") for i in range(len(test_filtered))]
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

    # Print sample ground truths for sanity check
    print("\nSample ground truths from filtered set:")
    for i in range(min(10, len(train_dataset))):
        gt = train_dataset[i]["reward_model"]["ground_truth"]
        q = train_dataset[i]["extra_info"]["question"][:80]
        print(f"  [{i}] GT: {gt:>30s}  Q: {q}...")

    print("Done.")


if __name__ == "__main__":
    main()
