#!/usr/bin/env python3
"""
Convert existing GSM8K parquet files to guidelines format
Reads existing train.parquet and test.parquet and converts them to the format
needed for guidelines training.
"""

import re
import os
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

def convert_dataset(input_path, output_path, split_name):
    """Convert a parquet file to guidelines format"""
    print(f"Loading {input_path}...")
    
    # Load existing parquet
    dataset = datasets.load_dataset("parquet", data_files=input_path)["train"]
    
    print(f"Found {len(dataset)} examples")
    print(f"Sample keys: {list(dataset[0].keys())}")
    
    def process_example(example, idx):
        # Handle different possible column names
        if "prompt" in example:
            question_raw = example["prompt"]
        elif "question" in example:
            question_raw = example["question"]
        else:
            raise ValueError(f"Could not find 'prompt' or 'question' in example. Keys: {list(example.keys())}")
        
        # Get ground truth
        if "ground_truth" in example:
            answer_raw = example.get("full_solution", example["ground_truth"])
            solution = extract_solution(str(answer_raw))
        elif "answer" in example:
            answer_raw = example["answer"]
            solution = extract_solution(str(answer_raw))
        elif "full_solution" in example:
            answer_raw = example["full_solution"]
            solution = extract_solution(str(answer_raw))
        else:
            raise ValueError(f"Could not find 'ground_truth', 'answer', or 'full_solution' in example. Keys: {list(example.keys())}")
        
        # Format prompt with system message
        question = SYSTEM_PROMPT + "\n\n" + "Question: " + question_raw
        
        # Create data in guidelines format
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
                "answer": answer_raw if isinstance(answer_raw, str) else str(answer_raw),
                "question": question_raw,
            },
        }
        return data
    
    # Process all examples
    print("Processing examples...")
    processed_data = [process_example(dataset[i], i) for i in range(len(dataset))]
    
    # Create new dataset
    new_dataset = Dataset.from_list(processed_data)
    
    # Save
    print(f"Saving to {output_path}...")
    new_dataset.to_parquet(output_path)
    
    print(f"✓ Converted {len(new_dataset)} examples")
    print(f"  Sample data_source: {new_dataset[0]['data_source']}")
    print(f"  Sample extra_info keys: {list(new_dataset[0]['extra_info'].keys())}")
    print(f"  Sample reward_model: {new_dataset[0]['reward_model']}")

if __name__ == "__main__":
    import datetime
    data_dir = "/projects/bfgx/jparekh/query-conditioned-guidelines/data/gsm8k"
    
    train_path = os.path.join(data_dir, "train.parquet")
    test_path = os.path.join(data_dir, "test.parquet")
    
    # Backup old files
    backup_train = None
    backup_test = None
    
    if os.path.exists(train_path):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_train = train_path + f".backup.{timestamp}"
        print(f"Backing up {train_path} to {backup_train}")
        os.rename(train_path, backup_train)
    else:
        print(f"ERROR: {train_path} does not exist!")
        exit(1)
    
    if os.path.exists(test_path):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_test = test_path + f".backup.{timestamp}"
        print(f"Backing up {test_path} to {backup_test}")
        os.rename(test_path, backup_test)
    else:
        print(f"ERROR: {test_path} does not exist!")
        exit(1)
    
    # Convert
    print("\n" + "="*60)
    print("Converting training set...")
    print("="*60)
    convert_dataset(backup_train, train_path, "train")
    
    print("\n" + "="*60)
    print("Converting test set...")
    print("="*60)
    convert_dataset(backup_test, test_path, "test")
    
    print("\n" + "="*60)
    print("✓ Conversion complete!")
    print("="*60)

