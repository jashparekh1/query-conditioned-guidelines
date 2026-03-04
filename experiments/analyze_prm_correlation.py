#!/usr/bin/env python3
"""Analyze correlation between min(PRM) and correctness from PRM scores JSON."""
import argparse
import json


def pearson(x, y):
    """Point-biserial correlation (Pearson between binary and continuous)."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    sx = (sum((a - mx) ** 2 for a in x) / n) ** 0.5
    sy = (sum((b - my) ** 2 for b in y) / n) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="PRM scores JSON from score_with_prm.py")
    args = parser.parse_args()

    data = json.load(open(args.input))

    all_min_prm = []
    all_correct = []
    all_mean_prm = []

    for q in data:
        for r in q["rollouts"]:
            all_min_prm.append(r["min_score"])
            all_mean_prm.append(r["mean_score"])
            all_correct.append(1.0 if r["correct"] else 0.0)

    n = len(all_correct)
    n_correct = int(sum(all_correct))
    print(f"Total rollouts: {n}")
    print(f"Correct: {n_correct}/{n} ({100 * n_correct / n:.1f}%)")
    print()

    r_min = pearson(all_min_prm, all_correct)
    r_mean = pearson(all_mean_prm, all_correct)

    print(f"Correlation(min_PRM, correctness) = {r_min:.4f}")
    print(f"Correlation(mean_PRM, correctness) = {r_mean:.4f}")
    print()

    # Per-question breakdown
    print("Per-question analysis:")
    print(f"  Q   acc   min_PRM(correct)  min_PRM(wrong)     gap")
    print("-" * 60)
    for q in data:
        correct_mins = [r["min_score"] for r in q["rollouts"] if r["correct"]]
        wrong_mins = [r["min_score"] for r in q["rollouts"] if not r["correct"]]
        avg_c = sum(correct_mins) / len(correct_mins) if correct_mins else 0
        avg_w = sum(wrong_mins) / len(wrong_mins) if wrong_mins else 0
        gap = avg_c - avg_w if correct_mins and wrong_mins else float("nan")
        print(f"  Q{q['question_idx']:>2} {q['accuracy']:>5.1f} {avg_c:>18.3f} {avg_w:>16.3f} {gap:>8.3f}")

    # min_PRM spread across rollouts (GRPO signal)
    print()
    print("GRPO signal (min_PRM spread per question):")
    for q in data:
        mins = [r["min_score"] for r in q["rollouts"]]
        spread = max(mins) - min(mins)
        print(f"  Q{q['question_idx']}: spread={spread:.4f} | mins={[round(m, 3) for m in mins]}")


if __name__ == "__main__":
    main()
