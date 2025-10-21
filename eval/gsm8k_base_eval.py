import argparse
import json
import os
from typing import List, Dict, Any

import datasets
from openai import OpenAI
from verl.utils.reward_score.gsm8k import extract_solution as rm_extract_solution

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# 与 examples/data_preprocess/gsm8k.py 对齐的指令
INSTRUCTION_FOLLOWING = 'Let\'s think step by step and output the final answer after "####".'


def normalize_number_text(text: str) -> str:
    return text.replace(",", "").strip()


def grade_answer(pred: str, gt: str) -> bool:
    if gt is None:
        return False
    gt_norm = normalize_number_text(gt)

    # 先用严格解析（#### number），再回退到更灵活的解析
    answer = rm_extract_solution(solution_str=pred, method="strict")
    if answer is None:
        answer = rm_extract_solution(solution_str=pred, method="flexible")
    if answer is None:
        return False
    return normalize_number_text(answer) == gt_norm


def build_openai_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))


def chat_complete(client: OpenAI, model: str, messages: List[Dict[str, Any]], temperature: float = 0.0, max_tokens: int = 1024) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=os.path.expanduser("~/data/gsm8k"))
    parser.add_argument("--split", default="test", choices=["train", "test"])  # evaluate test by default
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base_url", default="http://127.0.0.1:1225/v1")
    parser.add_argument("--model_name", default="base-llm")
    parser.add_argument("--output_path", default="gsm8k_base_eval_results.jsonl")
    args = parser.parse_args()

    # Load dataset parquet/json or HF repo
    if os.path.isdir(args.dataset_path):
        parquet_train = os.path.join(args.dataset_path, "train.parquet")
        parquet_test = os.path.join(args.dataset_path, "test.parquet")
        json_mix = os.path.join(args.dataset_path, "dataset.json")
        if os.path.exists(json_mix):
            with open(json_mix, "r", encoding="utf-8") as f:
                bundle = json.load(f)
            data = bundle[args.split]
            dataset = datasets.Dataset.from_list(data)
        elif os.path.exists(parquet_train) and os.path.exists(parquet_test):
            dataset_all = datasets.load_dataset("parquet", data_files={
                "train": parquet_train,
                "test": parquet_test,
            })
            dataset = dataset_all[args.split]
        else:
            dataset_all = datasets.load_dataset(args.dataset_path, "main")
            dataset = dataset_all[args.split]
    else:
        dataset_all = datasets.load_dataset(args.dataset_path, "main")
        dataset = dataset_all[args.split]

    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    client = build_openai_client(args.base_url)

    total = 0
    correct = 0

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for ex in tqdm(dataset, total=len(dataset), desc="GSM8K base eval"):
            # 读取问题与标准答案（兼容预处理/原始两种格式）
            messages: List[Dict[str, Any]]
            if "prompt" in ex:
                # 预处理格式：直接使用提供的 messages，GT 在 reward_model.ground_truth
                messages = ex.get("prompt", []) or []
                gt = ex.get("reward_model", {}).get("ground_truth")
                # 若缺失则尝试从 extra_info.answer 回退解析
                if (gt is None or gt == "") and ex.get("extra_info", {}).get("answer"):
                    gt = rm_extract_solution(ex.get("extra_info", {}).get("answer"), method="strict") or ""
                # 用于记录：尽量写入原始 question
                question_for_log = ex.get("extra_info", {}).get("question")
                if not question_for_log:
                    try:
                        user_msgs = [m for m in messages if m.get("role") == "user"]
                        question_candidate = user_msgs[-1]["content"] if user_msgs else ""
                    except Exception:
                        question_candidate = ""
                    suffix = f" {INSTRUCTION_FOLLOWING}"
                    if question_candidate.endswith(suffix):
                        question_for_log = question_candidate[: -len(suffix)]
                    else:
                        question_for_log = question_candidate
            else:
                # 原始 HF gsm8k：构造单条 user 消息，将指令拼接到问题后
                question_for_log = ex.get("question", "")
                messages = [
                    {"role": "user", "content": f"{question_for_log} {INSTRUCTION_FOLLOWING}"}
                ]
                answer_raw = ex.get("answer", "")
                gt = rm_extract_solution(answer_raw, method="strict") or ""

            total += 1

            # 单模型求解
            solution = chat_complete(
                client, args.model_name, messages, temperature=0.0, max_tokens=1024
            )

            is_correct = grade_answer(solution, gt)
            if is_correct:
                correct += 1

            rec = {
                "question": question_for_log,
                "solution": solution,
                "ground_truth": gt,
                "correct": bool(is_correct),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    acc = correct / max(1, total)
    print(json.dumps({"total": total, "correct": correct, "accuracy": acc}, ensure_ascii=False))


if __name__ == "__main__":
    main()


