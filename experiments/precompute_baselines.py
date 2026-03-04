"""
Pre-compute solver baseline accuracy WITHOUT guidelines for each training problem.

For each problem, runs the solver (14B-AWQ by default) with just the question
(no guideline) and records whether it gets the correct answer. This baseline is
used as the counterfactual in the reward function:

    R(guideline) = acc_with_guideline - baseline_acc

This makes the planner only get positive reward for genuinely helping the solver,
and negative reward for hurting it.

Usage:
    python -m experiments.precompute_baselines \
        --input experiments/data/numinamath_30k/train.parquet \
        --output experiments/data/numinamath_30k/baselines.json \
        --model Qwen/Qwen2.5-14B-Instruct-AWQ \
        --batch_size 64

Output: JSON file mapping question index -> baseline accuracy (0 or 1).
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from vllm import LLM, SamplingParams


SOLVER_SYSTEM_PROMPT_NO_GUIDELINE = """You are a careful and disciplined problem solver that reasons step by step and produces the final answer.

You are provided with a QUESTION that needs to be solved.

Your task:
- Think about the reasoning process as an internal monologue before giving the final answer.
- The reasoning process MUST be enclosed within <think> </think> tags.
- The final answer MUST be placed inside \\boxed{}.
- When performing reasoning, use precise and mathematical logic, not vague explanations.
- Do NOT reveal hidden reasoning outside of the <think> block.

Format:
<think>
Your detailed internal reasoning process, including any calculations, logic, or derivations.
</think>
\\boxed{your final answer here}"""


def build_messages(question: str):
    """Build chat messages for solver WITHOUT guideline."""
    return [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT_NO_GUIDELINE},
        {"role": "user", "content": f"Question: {question}\n\nPlease solve the question. Put reasoning inside <think> </think> and final answer inside \\boxed{{}}."},
    ]


def grade_answer(solver_output: str, ground_truth: str) -> float:
    """Grade solver output against ground truth. Returns 1.0 if correct, 0.0 otherwise."""
    # Try math_verify first (proper math equivalence)
    try:
        from verl.utils.reward_score.math_verify import compute_score
        return float(compute_score(solver_output, ground_truth))
    except Exception:
        pass

    # Fallback to mathruler
    try:
        from mathruler.grader import extract_boxed_content, grade_answer as mathruler_grade
        answer = extract_boxed_content(solver_output)
        return 1.0 if mathruler_grade(answer, ground_truth) else 0.0
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Pre-compute solver baselines without guidelines")
    parser.add_argument("--input", required=True, help="Path to train.parquet")
    parser.add_argument("--output", required=True, help="Path to output baselines JSON")
    parser.add_argument("--model", default=None, help="Solver model (default: VERL_GUIDELINES_SOLVER_MODEL or Qwen/Qwen2.5-14B-Instruct-AWQ)")
    parser.add_argument("--batch_size", type=int, default=64, help="vLLM batch size")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Max model length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--resume", action="store_true", help="Resume from partial output")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index (inclusive) for sharding")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (exclusive) for sharding")
    args = parser.parse_args()

    model_path = args.model or os.environ.get("VERL_GUIDELINES_SOLVER_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")

    # Load data
    print(f"Loading data from {args.input}")
    df = pd.read_parquet(args.input)
    questions = [row["question"] for row in df["extra_info"]]
    ground_truths = [row["ground_truth"] for row in df["reward_model"]]
    n_total = len(questions)
    print(f"Total problems: {n_total}")

    # Shard support: restrict to a slice of indices
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else n_total
    shard_indices = list(range(start_idx, end_idx))
    print(f"Shard: indices [{start_idx}, {end_idx}) = {len(shard_indices)} problems")

    # Resume support
    baselines = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r") as f:
            baselines = json.load(f)
        # Keys are strings in JSON
        baselines = {int(k): v for k, v in baselines.items()}
        print(f"Resuming: {len(baselines)} already computed")

    # Find remaining indices within this shard
    remaining = [i for i in shard_indices if i not in baselines]
    if not remaining:
        print("All baselines already computed!")
        return

    print(f"Computing baselines for {len(remaining)} problems")

    # Auto-detect quantization
    quantization = None
    model_lower = model_path.lower()
    if "gptq" in model_lower:
        quantization = "gptq"
    elif "awq" in model_lower:
        quantization = "awq_marlin"

    # Initialize vLLM
    os.environ["VLLM_USE_V1"] = "0"
    print(f"Loading model: {model_path} (quantization={quantization})")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="half" if quantization else "float16",
        quantization=quantization,
        trust_remote_code=True,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=3072, top_p=1.0)

    # Process in batches
    t0 = time.time()
    for batch_start in range(0, len(remaining), args.batch_size):
        batch_indices = remaining[batch_start:batch_start + args.batch_size]
        batch_questions = [questions[i] for i in batch_indices]
        batch_gts = [ground_truths[i] for i in batch_indices]

        # Build messages
        batch_messages = [build_messages(q) for q in batch_questions]

        # Run inference
        try:
            if hasattr(llm, "chat"):
                outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params, use_tqdm=True)
            else:
                prompts = [f"{SOLVER_SYSTEM_PROMPT_NO_GUIDELINE}\n\nQuestion: {q}" for q in batch_questions]
                outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        except Exception as e:
            print(f"ERROR: Batch inference failed: {e}")
            outputs = [None] * len(batch_messages)

        # Grade
        for idx, output, gt in zip(batch_indices, outputs, batch_gts):
            if output is None or not output.outputs:
                baselines[idx] = 0.0
                continue
            solver_output = output.outputs[0].text or ""
            baselines[idx] = grade_answer(solver_output, gt)

        # Save checkpoint
        elapsed = time.time() - t0
        done = batch_start + len(batch_indices)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - done) / rate if rate > 0 else 0
        n_correct = sum(1 for i in batch_indices if baselines.get(i, 0) == 1.0)
        print(f"[{done}/{len(remaining)}] batch acc={n_correct}/{len(batch_indices)} "
              f"({n_correct/len(batch_indices):.1%}) | "
              f"overall acc={sum(v for v in baselines.values())}/{len(baselines)} "
              f"({sum(v for v in baselines.values())/len(baselines):.1%}) | "
              f"ETA: {eta/60:.0f}min")

        # Save after every batch (resume-safe)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({str(k): v for k, v in sorted(baselines.items())}, f)

    total_time = time.time() - t0
    overall_acc = sum(v for v in baselines.values()) / len(baselines)
    print(f"\nDone! {len(baselines)} problems in {total_time/60:.1f}min")
    print(f"Solver baseline accuracy (no guideline): {overall_acc:.3f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
