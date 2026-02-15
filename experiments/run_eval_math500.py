#!/usr/bin/env python3
"""
Evaluate guilder + solver on MATH-500 using MathVerify for grading.
Uses the same prompts as training (experiments.prompts) and the same model for both guilder and solver.
Run from repo root: PYTHONPATH=. python -m experiments.run_eval_math500 [args]
Or: python experiments/run_eval_math500.py (adds project root to path automatically).
"""

import argparse
import json
import os
import sys

# Add project root so experiments and verl are importable when run as script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

from datasets import load_dataset
from vllm import LLM, SamplingParams

from experiments.prompts import GUILDER_SYSTEM_PROMPT, SOLVER_SYSTEM_PROMPT

# MathVerify: full solution string vs ground-truth answer
try:
    from verl.utils.reward_score.math_verify import compute_score as math_verify_score
except ImportError:
    math_verify_score = None


def write_jsonl(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on MATH-500 with MathVerify")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Model path for both guilder and solver")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="experiments/outputs/math500")
    parser.add_argument("--guilder_max_new_tokens", type=int, default=512)
    parser.add_argument("--solver_max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if math_verify_score is None:
        raise RuntimeError("MathVerify not available. Install with: pip install math-verify")

    os.makedirs(args.out_dir, exist_ok=True)
    guideline_path = os.path.join(args.out_dir, "math500_guidelines.jsonl")
    pred_path = os.path.join(args.out_dir, "math500_predictions.jsonl")

    print("Loading HuggingFaceH4/MATH-500...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split=args.split)
    if args.max_samples is not None and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    questions = [ex["problem"] for ex in ds]
    gold_answers = [ex["answer"] for ex in ds]
    print(f"Loaded {len(questions)} examples.")

    # ---------- Stage 1: Guilder ----------
    # Match training prompt format: single user message = GUILDER_SYSTEM_PROMPT + "\n\nQuestion: " + problem
    print("========== Stage 1: Guilder ==========")
    guilder_messages = [
        [{"role": "user", "content": GUILDER_SYSTEM_PROMPT + "\n\nQuestion: " + q}]
        for q in questions
    ]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_num_seqs=128,
        max_model_len=8192,
    )
    sampling = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=args.guilder_max_new_tokens)

    all_guidelines = []
    for start in range(0, len(guilder_messages), args.batch_size):
        batch = guilder_messages[start : start + args.batch_size]
        outputs = llm.chat(messages=batch, sampling_params=sampling, use_tqdm=False)
        for out in outputs:
            all_guidelines.append(out.outputs[0].text.strip())

    write_jsonl(guideline_path, [{"index": i, "question": q, "guideline": g} for i, (q, g) in enumerate(zip(questions, all_guidelines))])
    print(f"Saved guidelines to {guideline_path}")

    # ---------- Stage 2: Solver ----------
    print("========== Stage 2: Solver ==========")
    solver_messages = []
    for q, g in zip(questions, all_guidelines):
        user_content = (
            f"QUESTION:\n{q}\n\n"
            f"GUIDELINE:\n{g}\n\n"
            "Please strictly follow the GUIDELINE to solve the QUESTION. "
            "Remember to put your reasoning inside <think> </think> and final answer inside \\boxed{}."
        )
        solver_messages.append([
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ])

    sampling = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=args.solver_max_new_tokens)
    all_outputs = []
    for start in range(0, len(solver_messages), args.batch_size):
        batch = solver_messages[start : start + args.batch_size]
        outputs = llm.chat(messages=batch, sampling_params=sampling, use_tqdm=False)
        for out in outputs:
            all_outputs.append(out.outputs[0].text.strip())

    # ---------- Score with MathVerify ----------
    print("========== MathVerify scoring ==========")
    scores = []
    pred_rows = []
    for idx, (q, g, gold, pred_raw) in enumerate(zip(questions, all_guidelines, gold_answers, all_outputs)):
        # MATH-500 answer can be LaTeX; MathVerify expects model full output and ground-truth answer string
        try:
            s = math_verify_score(pred_raw, gold)
        except Exception:
            s = 0.0
        scores.append(s)
        pred_rows.append({
            "index": idx,
            "question": q,
            "guideline": g,
            "gold_answer": gold,
            "pred_raw": pred_raw,
            "math_verify_score": s,
        })

    write_jsonl(pred_path, pred_rows)
    print(f"Saved predictions to {pred_path}")

    accuracy = sum(scores) / len(scores) if scores else 0.0
    print(f"[MATH-500] #examples = {len(scores)}")
    print(f"[MATH-500] MathVerify accuracy = {accuracy * 100:.2f}%")

    summary_path = os.path.join(args.out_dir, "math500_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"num_examples: {len(scores)}\n")
        f.write(f"math_verify_accuracy: {accuracy:.6f}\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
