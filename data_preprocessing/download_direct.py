#!/usr/bin/env python3
"""
Download GSM8K directly and process it without using cache
"""

import os
import re
import datasets
from datasets import Dataset

SYSTEM_PROMPT = """You are a reasoning planner that creates structured, step-by-step guidelines for solving a given problem.
Your goal is not to answer the question directly, but to produce a high-quality, explicit plan that guides another model to solve it.
Each plan should:
1. Analyze what the problem is asking.
2. Identify the required knowledge, sub-tasks, or reasoning steps.
3. Provide a structured outline or set of instructions to follow.

Format your output as a concise, ordered list of reasoning steps or directives. Avoid giving the final answer."""

def extract_solution(solution_str):
    """Extract final answer from GSM8K format"""
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    if solution is None:
        # Fallback: try to extract any number at the end
        numbers = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        if numbers:
            return numbers[-1].replace(",", "")
        return solution_str.strip()
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def process_dataset(dataset, split_name):
    """Process dataset to guidelines format"""
    def process_fn(example, idx):
        question_raw = example["question"]
        question = SYSTEM_PROMPT + "\n\n" + "Question: " + question_raw
        
        answer_raw = example["answer"]
        solution = extract_solution(answer_raw)
        
        data = {
            "data_source": "guidelines",
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split_name,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            },
        }
        return data
    
    processed = [process_fn(dataset[i], i) for i in range(len(dataset))]
    return Dataset.from_list(processed)

if __name__ == "__main__":
    data_dir = "/projects/bfgx/jparekh/query-conditioned-guidelines/data/gsm8k"
    os.makedirs(data_dir, exist_ok=True)
    
    # Backup old files
    train_path = os.path.join(data_dir, "train.parquet")
    test_path = os.path.join(data_dir, "test.parquet")
    
    if os.path.exists(train_path):
        import datetime
        backup_train = train_path + f".backup.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Backing up {train_path} to {backup_train}")
        os.rename(train_path, backup_train)
    
    if os.path.exists(test_path):
        import datetime
        backup_test = test_path + f".backup.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Backing up {test_path} to {backup_test}")
        os.rename(test_path, backup_test)
    
    # Download directly in memory without cache
    print(f"Downloading GSM8K from HuggingFace (this may take a minute)...")
    print("Downloading directly in memory (no cache)...")
    
    # Use /tmp which has 1.5T available (instead of /projects which is full)
    import tempfile
    tempfile.tempdir = "/tmp"  # Force tempfile to use /tmp
    with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
        # Set cache to temp directory (will be deleted after)
        os.environ['HF_DATASETS_CACHE'] = temp_dir
        os.environ['HF_HOME'] = temp_dir
        
        # Download dataset
        dataset = datasets.load_dataset("openai/gsm8k", "main", download_mode="reuse_cache_if_exists")
        
        # Process immediately while dataset is in memory
        print(f"Downloaded {len(dataset['train'])} train and {len(dataset['test'])} test examples")
        
        # Process training set
        print("\nProcessing training set...")
        train_processed = process_dataset(dataset["train"], "train")
        print(f"Saving to {train_path}...")
        train_processed.to_parquet(train_path)
        print(f"✓ Saved {len(train_processed)} training examples")
        
        # Process test set
        print("\nProcessing test set...")
        test_processed = process_dataset(dataset["test"], "test")
        print(f"Saving to {test_path}...")
        test_processed.to_parquet(test_path)
        print(f"✓ Saved {len(test_processed)} test examples")
        
        # Clear dataset from memory
        del dataset
        del train_processed
        del test_processed
    
    # Verify by loading the saved files
    print("\n" + "="*60)
    print("Verification:")
    print("="*60)
    train_verify = datasets.load_dataset("parquet", data_files=train_path)["train"]
    print(f"Train examples: {len(train_verify)}")
    print(f"Sample train keys: {list(train_verify[0].keys())}")
    print(f"Sample train data_source: {train_verify[0]['data_source']}")
    print(f"Sample train extra_info keys: {list(train_verify[0]['extra_info'].keys())}")
    print(f"Sample train reward_model: {train_verify[0]['reward_model']}")
    print("="*60)
    print("\n✓ All done! Data is ready for training.")

