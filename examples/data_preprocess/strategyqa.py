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
Preprocess the StrategyQA dataset to parquet format
"""

import argparse
import os
import json
import shutil

import datasets


SYSTEM_PROMPT = """You are a reasoning planner that creates structured, step-by-step guidelines for answering a strategic reasoning question.
Your goal is not to answer the question directly, but to produce a high-quality, explicit plan that guides another model to solve it.
These questions require implicit multi-hop reasoning and the ability to connect facts that are not explicitly stated.
Each plan should:
1. Analyze what the question is asking and what implicit knowledge is required.
2. Break down the question into component facts or sub-questions that need to be verified.
3. Identify the reasoning strategy needed to connect these facts.
4. Provide a structured approach to reach a yes/no conclusion.
5. Consider potential edge cases or alternative interpretations.

Format your output as a concise, ordered list of reasoning steps or directives. Avoid giving the final answer."""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default="../../data/strategyqa", help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/strategyqa", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "strategyqa"

    if local_dataset_path is not None:
        dataset = datasets.load_from_disk(local_dataset_path)
    else:
        dataset = datasets.load_dataset("ChilleD/StrategyQA")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["question"]
            answer = example["answer"]
            
            # Convert boolean to string
            answer_str = "Yes" if answer else "No"
            
            question = SYSTEM_PROMPT + "\n\n" + "Question: " + question_raw + "\n\nAnswer with Yes or No."

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "strategic_reasoning",
                "reward_model": {"style": "rule", "ground_truth": answer_str},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question_raw,
                    "answer": answer,
                    "qid": example.get("qid", f"{split}_{idx}"),
                    "term": example.get("term", ""),
                    "description": example.get("description", ""),
                    "facts": example.get("facts", "")
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

