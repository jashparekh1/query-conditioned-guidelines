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
Preprocess the CommonsenseQA dataset to parquet format
"""

import argparse
import os
import json
import shutil

import datasets


SYSTEM_PROMPT = """You are a reasoning planner that creates structured, step-by-step guidelines for solving a commonsense reasoning question.
Your goal is not to answer the question directly, but to produce a high-quality, explicit plan that guides another model to solve it.
Each plan should:
1. Analyze what the question is asking and what commonsense knowledge is required.
2. Identify the key concepts and relationships between the answer choices.
3. Provide a structured reasoning process to evaluate each option.
4. Guide the model to select the most appropriate answer based on commonsense reasoning.

Format your output as a concise, ordered list of reasoning steps or directives. Avoid giving the final answer."""


def format_choices(choices):
    """Format the multiple choice options into a readable string."""
    labels = choices['label']
    texts = choices['text']
    formatted = []
    for label, text in zip(labels, texts):
        formatted.append(f"({label}) {text}")
    return "\n".join(formatted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default="../../data/commonsenseqa", help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/commonsenseqa", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "commonsenseqa"

    if local_dataset_path is not None:
        dataset = datasets.load_from_disk(local_dataset_path)
    else:
        dataset = datasets.load_dataset("tau/commonsense_qa")

    train_dataset = dataset["train"]
    # CommonsenseQA has validation and test splits
    validation_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["question"]
            choices_raw = example["choices"]
            answer_key = example["answerKey"]
            
            # Format the question with choices
            choices_formatted = format_choices(choices_raw)
            question = SYSTEM_PROMPT + "\n\n" + "Question: " + question_raw + "\n\nAnswer Choices:\n" + choices_formatted

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "commonsense_reasoning",
                "reward_model": {"style": "rule", "ground_truth": answer_key},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question_raw,
                    "choices": choices_raw,
                    "answer_key": answer_key,
                    "id": example.get("id", f"{split}_{idx}"),
                    "question_concept": example.get("question_concept", "")
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    validation_dataset = validation_dataset.map(function=make_map_fn("validation"), with_indices=True)
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
    validation_dataset.to_parquet(os.path.join(local_save_dir, "validation.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    # Save all datasets into a single JSON file
    combined_output = {
        "train": list(train_dataset),
        "validation": list(validation_dataset),
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

