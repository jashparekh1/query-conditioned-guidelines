# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH dataset to parquet format
"""

import argparse
import os
import re
import json
import shutil

import datasets


def extract_boxed_answer(solution_str):
    """Extract the final answer from \\boxed{} notation."""
    # Try to find the last \boxed{} in the solution
    matches = re.findall(r'\\boxed\{([^}]*)\}', solution_str)
    if matches:
        return matches[-1]  # Return the last boxed answer
    return solution_str.strip()


SYSTEM_PROMPT = """You are a reasoning planner that creates structured, step-by-step guidelines for solving a challenging mathematics problem.
Your goal is not to solve the problem directly, but to produce a high-quality, explicit plan that guides another model to solve it.
Each plan should:
1. Analyze what the problem is asking and identify the key mathematical concepts involved.
2. Break down the problem into manageable sub-problems or steps.
3. Identify the mathematical techniques, formulas, or theorems that might be useful.
4. Provide a structured outline of the solution approach.
5. Highlight any potential pitfalls or special cases to consider.

Format your output as a concise, ordered list of reasoning steps or directives. Avoid giving the final answer or complete solution."""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default="/shared/nas2/heng6/course/query-conditioned-guidelines/data/math", help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/math", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "math"

    if local_dataset_path is not None:
        dataset = datasets.load_from_disk(local_dataset_path)
    else:
        # If loading from scratch, need to load and combine all categories
        from datasets import DatasetDict, concatenate_datasets
        categories = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 
                      'number_theory', 'prealgebra', 'precalculus']
        all_train = []
        all_test = []
        for category in categories:
            ds = datasets.load_dataset('EleutherAI/hendrycks_math', category)
            all_train.append(ds['train'])
            all_test.append(ds['test'])
        dataset = DatasetDict({
            'train': concatenate_datasets(all_train),
            'test': concatenate_datasets(all_test)
        })

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem_raw = example["problem"]
            solution_raw = example["solution"]
            level = example.get("level", "")
            problem_type = example.get("type", "")

            question = SYSTEM_PROMPT + "\n\n" + "Problem: " + problem_raw

            # Extract the final answer
            final_answer = extract_boxed_answer(solution_raw)
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "advanced_math",
                "reward_model": {"style": "rule", "ground_truth": final_answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "problem": problem_raw,
                    "solution": solution_raw,
                    "level": level,
                    "type": problem_type,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir
    
    # Expand ~ in path
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    # Save both datasets into a single JSON file
    combined_output = {
        "train": list(train_dataset),
        "test": list(test_dataset),
    }
    with open(os.path.join(local_save_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(combined_output, f, ensure_ascii=False)

    if hdfs_dir is not None:
        os.makedirs(hdfs_dir, exist_ok=True)
        # Copy directory contents
        if os.path.exists(hdfs_dir):
            shutil.rmtree(hdfs_dir)
        shutil.copytree(local_save_dir, hdfs_dir)

