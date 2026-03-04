#!/usr/bin/env python3
"""
Debug script: run full planner→solver pipeline on N questions, K rollouts each.
Saves FULL (no truncation) guidelines + solver outputs to JSON + prints everything.

Usage:
    python -m experiments.debug_trajectories [--n 10] [--rollouts 4] [--seed 123]
"""
import argparse
import json
import os
import re

import pandas as pd


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks and \\boxed{...} from planner output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\\boxed\{[^}]*\}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text



def extract_boxed(text):
    """Extract content from last \\boxed{...} (nested-brace aware)."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i-1].strip() if depth == 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of questions")
    parser.add_argument("--rollouts", type=int, default=4, help="Rollouts per question")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Planner model")
    parser.add_argument("--solver", type=str, default="Qwen/Qwen2.5-14B-Instruct-AWQ", help="Solver model")
    parser.add_argument("--data", type=str, default="experiments/data/numinamath_30k/train.parquet")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=1.0, help="Planner sampling temperature")
    parser.add_argument("--out", type=str, default="experiments/logs/debug_trajectories_full.json",
                        help="Output JSON (FULL, no truncation)")
    args = parser.parse_args()

    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM, SamplingParams

    # --- Load models ---
    def make_llm(model_path, gpu_mem=0.45):
        quantization = None
        dtype = "float16"
        if "awq" in model_path.lower():
            quantization = "awq_marlin"
            dtype = "half"
        elif "gptq" in model_path.lower():
            quantization = "gptq"
            dtype = "half"
        print(f"Loading model: {model_path} (quantization={quantization})")
        return LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_mem,
            max_model_len=8192,
            dtype=dtype,
            quantization=quantization,
            trust_remote_code=True,
            enforce_eager=True,
        )

    use_separate = args.model != args.solver
    planner_llm = make_llm(args.model, gpu_mem=0.40 if use_separate else 0.85)
    solver_llm = make_llm(args.solver, gpu_mem=0.40) if use_separate else planner_llm

    # --- Load data ---
    df = pd.read_parquet(args.data)
    sample = df.sample(n=args.n, random_state=args.seed)

    from experiments.prompts import PLANNER_SYSTEM_PROMPT, SOLVER_SYSTEM_PROMPT

    planner_sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, top_p=1.0)
    solver_sampling = SamplingParams(temperature=0.0, max_tokens=3072, top_p=1.0)

    # --- Grading setup ---
    try:
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        def grade(solver_text, gt):
            try:
                score, _ = verify_func(["\\boxed{" + gt + "}"], [solver_text])
                return bool(score)
            except Exception:
                return False
    except ImportError:
        print("WARNING: math_verify not available, using string match")
        from mathruler.grader import extract_boxed_content, grade_answer
        def grade(solver_text, gt):
            return grade_answer(extract_boxed_content(solver_text), gt)

    # --- Run pipeline ---
    all_results = []

    for q_idx, (_, row) in enumerate(sample.iterrows()):
        question = row["extra_info"]["question"]
        ground_truth = row["reward_model"]["ground_truth"]

        print(f"\n{'#'*100}")
        print(f"# QUESTION {q_idx+1}/{args.n}")
        print(f"{'#'*100}")
        print(f"\nQUESTION: {question}")
        print(f"GROUND TRUTH: {ground_truth}")

        # Generate K planner rollouts in one batch
        planner_messages_batch = [
            [{"role": "user", "content": PLANNER_SYSTEM_PROMPT + "\n\nQuestion: " + question}]
            for _ in range(args.rollouts)
        ]
        planner_outputs = planner_llm.chat(
            messages=planner_messages_batch,
            sampling_params=planner_sampling,
            use_tqdm=False,
        )

        # For each rollout, run solver
        question_results = []
        for r_idx, p_out in enumerate(planner_outputs):
            raw_guideline = p_out.outputs[0].text if p_out.outputs else ""
            clean_guideline = strip_think_tags(raw_guideline)
            has_think = "<think>" in raw_guideline.lower()
            has_boxed = "\\boxed{" in raw_guideline

            print(f"\n{'='*80}")
            print(f"  ROLLOUT {r_idx+1}/{args.rollouts}")
            print(f"{'='*80}")

            print(f"\n  --- RAW PLANNER OUTPUT ({len(raw_guideline)} chars) ---")
            print(raw_guideline)

            print(f"\n  --- CLEANED GUIDELINE sent to solver ({len(clean_guideline)} chars) ---")
            print(clean_guideline)

            # Run solver
            solver_user = (
                f"Question: {question}\n\nGuideline: {clean_guideline}\n\n"
                "Please strictly follow the Guideline to solve the Question. "
                "Put reasoning inside <think> </think> and final answer inside \\boxed{}."
            )
            solver_messages = [
                {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": solver_user},
            ]
            solver_output = solver_llm.chat(
                messages=[solver_messages],
                sampling_params=solver_sampling,
                use_tqdm=False,
            )
            solver_text = solver_output[0].outputs[0].text if solver_output[0].outputs else ""

            print(f"\n  --- SOLVER OUTPUT ({len(solver_text)} chars) ---")
            print(solver_text)

            # Grade
            correct = grade(solver_text, ground_truth)
            solver_answer = extract_boxed(solver_text)

            print(f"\n  --- RESULT ---")
            print(f"  Solver answer: {solver_answer}")
            print(f"  Ground truth:  {ground_truth}")
            print(f"  Correct: {'YES' if correct else 'NO'}")

            question_results.append({
                "rollout": r_idx,
                "raw_guideline": raw_guideline,
                "clean_guideline": clean_guideline,
                "has_think": has_think,
                "has_boxed": has_boxed,
                "solver_text": solver_text,
                "solver_answer": solver_answer,
                "correct": correct,
            })

        # Summary for this question
        n_correct = sum(1 for r in question_results if r["correct"])
        print(f"\n  >>> QUESTION SUMMARY: {n_correct}/{args.rollouts} correct <<<")

        all_results.append({
            "question_idx": q_idx,
            "question": question,
            "ground_truth": ground_truth,
            "rollouts": question_results,
            "accuracy": n_correct / args.rollouts,
        })

    # --- Save full JSON (NO TRUNCATION) ---
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n\nSaved full results to {args.out}")

    # --- Overall summary ---
    print(f"\n{'#'*100}")
    print("OVERALL SUMMARY")
    print(f"{'#'*100}")
    for r in all_results:
        n_c = sum(1 for ro in r["rollouts"] if ro["correct"])
        print(f"  Q{r['question_idx']+1}: {n_c}/{args.rollouts} correct | GT={r['ground_truth']} | {r['question'][:80]}...")
    total_correct = sum(sum(1 for ro in r["rollouts"] if ro["correct"]) for r in all_results)
    total = args.n * args.rollouts
    print(f"\n  TOTAL: {total_correct}/{total} ({100*total_correct/total:.0f}%)")


if __name__ == "__main__":
    main()
