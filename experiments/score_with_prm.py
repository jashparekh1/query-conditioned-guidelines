#!/usr/bin/env python3
"""
Score solver outputs from a debug_trajectories JSON with Qwen2.5-Math-PRM-7B.
Check: do different guidelines → different PRM scores? Is there variance?

Usage:
    python -m experiments.score_with_prm \
        --input experiments/logs/debug_dapo100_4b_14b.json \
        --model Qwen/Qwen2.5-Math-PRM-7B
"""
import argparse
import json
import os
import re

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def split_solver_into_steps(solver_text: str) -> list[str]:
    """Split solver output into reasoning steps."""
    # Remove <think> tags but keep content
    text = re.sub(r"</?think>", "", solver_text).strip()
    # Split on double newlines, step markers, or bullet points
    steps = re.split(r"\n\n+|\n(?=\*\*|Step |\d+\.\s)", text)
    steps = [s.strip() for s in steps if s.strip()]
    return steps if steps else [text]


def make_step_rewards(logits, token_masks):
    """Extract step-level reward scores from PRM output."""
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


def score_with_prm(model, tokenizer, question: str, solver_text: str) -> dict:
    """Score a solver output step-by-step with the PRM."""
    steps = split_solver_into_steps(solver_text)

    # Build messages in the format the PRM expects
    # Steps joined with <extra_0> separator
    response_str = "<extra_0>".join(steps) + "<extra_0>"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response_str},
    ]

    conversation_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = tokenizer.encode(conversation_str, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)

    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)
    step_scores_batch = make_step_rewards(outputs[0], token_masks)
    step_scores = step_scores_batch[0] if step_scores_batch else []

    return {
        "step_scores": step_scores,
        "mean_score": sum(step_scores) / len(step_scores) if step_scores else 0.0,
        "min_score": min(step_scores) if step_scores else 0.0,
        "n_steps": len(step_scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to debug_trajectories JSON")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--out", default=None, help="Output JSON with PRM scores")
    args = parser.parse_args()

    if args.out is None:
        args.out = args.input.replace(".json", "_prm_scores.json")

    print(f"Loading PRM: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    print(f"Loading debug data from {args.input}")
    data = json.load(open(args.input))

    results = []
    for q_idx, item in enumerate(data):
        question = item["question"]
        gt = item["ground_truth"]
        print(f"\nQ{q_idx}: {question[:80]}... (GT={gt})")

        rollout_scores = []
        for r_idx, rollout in enumerate(item["rollouts"]):
            solver_text = rollout["solver_text"]
            correct = rollout["correct"]

            prm = score_with_prm(model, tokenizer, question, solver_text)
            print(f"  R{r_idx}: correct={correct} | PRM mean={prm['mean_score']:.3f} min={prm['min_score']:.3f} steps={prm['n_steps']}")

            rollout_scores.append({
                "rollout": r_idx,
                "correct": correct,
                "guideline_preview": rollout["clean_guideline"][:150],
                **prm,
            })

        # Variance analysis
        means = [r["mean_score"] for r in rollout_scores]
        if len(means) > 1:
            import statistics
            variance = statistics.variance(means)
            spread = max(means) - min(means)
        else:
            variance = 0.0
            spread = 0.0

        print(f"  >>> PRM score spread: {spread:.4f} | variance: {variance:.6f}")

        results.append({
            "question_idx": q_idx,
            "question": question[:150],
            "ground_truth": gt,
            "accuracy": item["accuracy"],
            "prm_score_variance": variance,
            "prm_score_spread": spread,
            "rollouts": rollout_scores,
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: PRM Score Variance Across Rollouts")
    print("=" * 80)
    for r in results:
        means = [ro["mean_score"] for ro in r["rollouts"]]
        print(f"  Q{r['question_idx']}: acc={r['accuracy']:.1f} | PRM means: {[round(m, 3) for m in means]} | spread={r['prm_score_spread']:.4f}")

    avg_spread = sum(r["prm_score_spread"] for r in results) / len(results)
    print(f"\n  Average PRM spread across questions: {avg_spread:.4f}")
    print(f"  (Need spread > 0.05 for meaningful GRPO signal)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
