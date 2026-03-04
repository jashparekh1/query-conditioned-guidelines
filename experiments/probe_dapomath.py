#!/usr/bin/env python3
"""
Probe DAPO-Math-17k: load N questions, run 14B-AWQ solver (no guideline),
report accuracy. Use this to decide if the dataset is the right difficulty.

Usage:
    python -m experiments.probe_dapomath [--n 200] [--solver Qwen/Qwen2.5-14B-Instruct-AWQ]
"""
import argparse
import os
import re

os.environ["VLLM_USE_V1"] = "0"


def extract_boxed(text):
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
    return text[start:i - 1].strip() if depth == 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--solver", type=str, default="Qwen/Qwen2.5-14B-Instruct-AWQ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="experiments/logs/probe_dapomath.json")
    args = parser.parse_args()

    import datasets
    import json

    print(f"Loading DAPO-Math-17k ...")
    ds = datasets.load_dataset(
        "BytedTsinghua-SIA/DAPO-Math-17k",
        split="train",
        cache_dir="/tmp/hf_cache",
    )
    print(f"Dataset size: {len(ds)}")

    # Sample N questions
    import random
    random.seed(args.seed)
    indices = random.sample(range(len(ds)), args.n)
    sample = [ds[i] for i in indices]

    # Extract question text and ground truth
    questions = []
    ground_truths = []
    for row in sample:
        # prompt is a list of dicts with 'content' and 'role'
        content = row["prompt"][0]["content"] if isinstance(row["prompt"], list) else row["prompt"]["content"]
        # Strip DAPO boilerplate. Full prefix format:
        # "Solve the following math problem step by step. The last line of your
        #  response should be of the form Answer: $Answer (without quotes) where
        #  $Answer is the answer to the problem.\n\n<PROBLEM>\n\nRemember to put
        #  your answer on its own line after "Answer:"."
        # Strip prefix up to the first double newline after the boilerplate
        content = re.sub(
            r"^Solve the following math problem step by step\..*?where \$Answer is the answer to the problem\.\s*",
            "",
            content.strip(),
            flags=re.DOTALL,
        )
        # Strip suffix
        content = re.sub(
            r"\s*Remember to put your answer on its own line after [\"']?Answer:[\"']?.*$",
            "",
            content,
            flags=re.DOTALL,
        )
        content = content.strip()
        gt = row["reward_model"]["ground_truth"]
        questions.append(content)
        ground_truths.append(gt)

    print(f"\nSample ground truths (first 10): {ground_truths[:10]}")
    print(f"Sample question: {questions[0][:200]}\n")

    # Load solver
    from vllm import LLM, SamplingParams

    print(f"Loading solver: {args.solver}")
    quantization = None
    dtype = "float16"
    if "awq" in args.solver.lower():
        quantization = "awq_marlin"
        dtype = "half"
    elif "gptq" in args.solver.lower():
        quantization = "gptq"
        dtype = "half"

    llm = LLM(
        model=args.solver,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        dtype=dtype,
        quantization=quantization,
        trust_remote_code=True,
        enforce_eager=True,
    )

    SOLVER_SYSTEM_PROMPT = """You are a careful math problem solver. Solve the problem step by step.
Put your reasoning inside <think> </think> tags and your final answer inside \\boxed{}."""

    sampling = SamplingParams(temperature=0.0, max_tokens=3072, top_p=1.0)

    messages = [
        [
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {q}\n\nSolve this problem."},
        ]
        for q in questions
    ]

    print(f"Running solver on {args.n} questions ...")
    outputs = llm.chat(messages=messages, sampling_params=sampling, use_tqdm=True)

    # Grade
    try:
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        def grade(pred_text, gt):
            try:
                score, _ = verify_func(["\\boxed{" + gt + "}"], [pred_text])
                return bool(score)
            except Exception:
                return False
    except ImportError:
        from mathruler.grader import extract_boxed_content, grade_answer
        def grade(pred_text, gt):
            return grade_answer(extract_boxed_content(pred_text), gt)

    results = []
    n_correct = 0
    n_no_answer = 0

    for i, (output, q, gt) in enumerate(zip(outputs, questions, ground_truths)):
        solver_text = output.outputs[0].text if output.outputs else ""
        correct = grade(solver_text, gt)
        answer = extract_boxed(solver_text)
        if answer is None:
            n_no_answer += 1

        if correct:
            n_correct += 1

        results.append({
            "question": q,
            "ground_truth": gt,
            "solver_answer": answer,
            "solver_text": solver_text,
            "correct": correct,
        })

        if i < 10 or (not correct and i < 30):
            print(f"\n[{i+1}] GT={gt} | Pred={answer} | {'✓' if correct else '✗'}")
            print(f"  Q: {q[:120]}...")

    accuracy = n_correct / args.n
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.solver}")
    print(f"  N={args.n}, Correct={n_correct}, Accuracy={accuracy:.1%}")
    print(f"  No answer (no \\boxed{{}}): {n_no_answer}")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "solver": args.solver,
            "n": args.n,
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_no_answer": n_no_answer,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
