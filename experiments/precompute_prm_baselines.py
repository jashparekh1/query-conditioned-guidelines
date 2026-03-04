"""
Pre-compute PRM baselines: min(PRM) for solver WITHOUT planner guidelines.

For each problem, runs the solver with just the question (no guideline),
scores the output with Qwen2.5-Math-PRM-7B, and records min(PRM step scores).

Used as counterfactual in PRM-based reward:
    R_planner = min(PRM_with_plan) - min(PRM_without_plan)

Usage:
    python -m experiments.precompute_prm_baselines \
        --input experiments/data/dapomath_30k/train.parquet \
        --output experiments/data/dapomath_30k/prm_baselines.json \
        --n 3000 --seed 42
"""

import argparse
import json
import os
import re
import time

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


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

PRM_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def build_messages(question: str):
    return [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT_NO_GUIDELINE},
        {"role": "user", "content": f"Question: {question}\n\nPlease solve the question. Put reasoning inside <think> </think> and final answer inside \\boxed{{}}."},
    ]


def split_solver_into_steps(solver_text: str) -> list:
    text = re.sub(r"</?think>", "", solver_text).strip()
    steps = re.split(r"\n\n+|\n(?=\*\*|Step |\d+\.\s)", text)
    steps = [s.strip() for s in steps if s.strip()]
    return steps if steps else [text]


def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    all_scores = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        all_scores.append(positive_probs.cpu().tolist())
    return all_scores


def score_with_prm(prm_model, prm_tokenizer, step_sep_id, question, solver_text):
    steps = split_solver_into_steps(solver_text)
    response_str = "<extra_0>".join(steps) + "<extra_0>"
    messages = [
        {"role": "system", "content": PRM_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response_str},
    ]
    conversation_str = prm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = prm_tokenizer.encode(conversation_str, return_tensors="pt").to(prm_model.device)
    with torch.no_grad():
        outputs = prm_model(input_ids=input_ids, use_cache=False)
    token_masks = (input_ids == step_sep_id)
    step_scores_batch = make_step_rewards(outputs[0], token_masks)
    step_scores = step_scores_batch[0] if step_scores_batch else []
    return {
        "min_score": min(step_scores) if step_scores else 0.0,
        "mean_score": sum(step_scores) / len(step_scores) if step_scores else 0.0,
        "n_steps": len(step_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-compute PRM baselines (solver without guidelines)")
    parser.add_argument("--input", required=True, help="Path to train.parquet")
    parser.add_argument("--output", required=True, help="Path to output prm_baselines.json")
    parser.add_argument("--solver", default=None, help="Solver model")
    parser.add_argument("--prm", default="Qwen/Qwen2.5-Math-PRM-7B", help="PRM model")
    parser.add_argument("--n", type=int, default=3000, help="Number of questions to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--batch_size", type=int, default=32, help="Solver vLLM batch size")
    parser.add_argument("--resume", action="store_true", help="Resume from partial output")
    args = parser.parse_args()

    solver_path = args.solver or os.environ.get("VERL_GUIDELINES_SOLVER_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")

    # Load data and sample
    print(f"Loading data from {args.input}")
    df = pd.read_parquet(args.input)
    questions = [row["question"] for row in df["extra_info"]]
    ground_truths = [row["ground_truth"] for row in df["reward_model"]]
    n_total = len(questions)
    print(f"Total problems: {n_total}")

    import random
    random.seed(args.seed)
    if args.n < n_total:
        sample_indices = sorted(random.sample(range(n_total), args.n))
    else:
        sample_indices = list(range(n_total))
    print(f"Processing {len(sample_indices)} questions")

    # Resume support
    baselines = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r") as f:
            baselines = {int(k): v for k, v in json.load(f).items()}
        print(f"Resuming: {len(baselines)} already computed")

    remaining = [i for i in sample_indices if i not in baselines]
    if not remaining:
        print("All PRM baselines already computed!")
        return

    print(f"Computing PRM baselines for {len(remaining)} problems")

    # ── Phase 1: Solver inference (vLLM) ──
    print(f"\n=== Phase 1: Solver inference ({solver_path}) ===")
    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM, SamplingParams

    quantization = None
    model_lower = solver_path.lower()
    if "gptq" in model_lower:
        quantization = "gptq"
    elif "awq" in model_lower:
        quantization = "awq_marlin"

    llm = LLM(
        model=solver_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        dtype="half" if quantization else "float16",
        quantization=quantization,
        trust_remote_code=True,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=3072, top_p=1.0)

    solver_outputs = {}  # idx -> solver_text
    t0 = time.time()
    for batch_start in range(0, len(remaining), args.batch_size):
        batch_indices = remaining[batch_start:batch_start + args.batch_size]
        batch_messages = [build_messages(questions[i]) for i in batch_indices]

        try:
            outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params, use_tqdm=True)
        except Exception as e:
            print(f"ERROR: Batch inference failed: {e}")
            outputs = [None] * len(batch_messages)

        for idx, output in zip(batch_indices, outputs):
            text = ""
            if output is not None and output.outputs:
                text = output.outputs[0].text or ""
            solver_outputs[idx] = text

        done = batch_start + len(batch_indices)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - done) / rate if rate > 0 else 0
        print(f"  Solver [{done}/{len(remaining)}] ETA: {eta / 60:.0f}min")

    # Free solver GPU memory before loading PRM
    del llm
    torch.cuda.empty_cache()
    print(f"Solver done. {len(solver_outputs)} outputs in {(time.time() - t0) / 60:.1f}min")

    # ── Phase 2: PRM scoring ──
    print(f"\n=== Phase 2: PRM scoring ({args.prm}) ===")
    prm_tokenizer = AutoTokenizer.from_pretrained(args.prm, trust_remote_code=True)
    prm_model = AutoModel.from_pretrained(
        args.prm,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    step_sep_id = prm_tokenizer.encode("<extra_0>")[0]

    t1 = time.time()
    for count, idx in enumerate(remaining):
        solver_text = solver_outputs.get(idx, "")
        if not solver_text.strip():
            baselines[idx] = 0.0
            continue

        try:
            prm = score_with_prm(prm_model, prm_tokenizer, step_sep_id, questions[idx], solver_text)
            baselines[idx] = {
                "min_score": prm["min_score"],
                "mean_score": prm["mean_score"],
                "n_steps": prm["n_steps"],
            }
        except Exception as e:
            print(f"  PRM error for idx {idx}: {e}")
            baselines[idx] = {"min_score": 0.0, "mean_score": 0.0, "n_steps": 0}

        if (count + 1) % 100 == 0:
            elapsed = time.time() - t1
            rate = (count + 1) / elapsed
            eta = (len(remaining) - count - 1) / rate
            done_scores = [baselines[i]["min_score"] for i in remaining[:count + 1] if isinstance(baselines.get(i), dict)]
            avg_min = sum(done_scores) / len(done_scores) if done_scores else 0
            print(f"  PRM [{count + 1}/{len(remaining)}] avg_min_PRM={avg_min:.3f} ETA: {eta / 60:.0f}min")

        # Save every 500
        if (count + 1) % 500 == 0:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump({str(k): v for k, v in sorted(baselines.items())}, f)

    # Final save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({str(k): v for k, v in sorted(baselines.items())}, f)

    total_time = time.time() - t0
    all_mins = [v["min_score"] for v in baselines.values() if isinstance(v, dict)]
    all_means = [v["mean_score"] for v in baselines.values() if isinstance(v, dict)]
    avg_min = sum(all_mins) / len(all_mins) if all_mins else 0
    avg_mean = sum(all_means) / len(all_means) if all_means else 0
    print(f"\nDone! {len(baselines)} PRM baselines in {total_time / 60:.1f}min")
    print(f"Avg min(PRM) without guidelines: {avg_min:.3f}")
    print(f"Avg mean(PRM) without guidelines: {avg_mean:.3f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
