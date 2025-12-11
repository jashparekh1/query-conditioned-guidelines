#!/usr/bin/env python3
"""
CommonsenseQA Baseline Evaluation Script
Evaluates a model's performance on the CommonsenseQA dataset and calculates accuracy metrics.
"""

import os
import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
import dotenv
import math
import time

from prompt_builder import PromptBuilder, COTPromptBuilder, ICLPromptBuilder

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

dotenv.load_dotenv()

try:
    import pandas as pd
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ Transformers version: {transformers.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("Please install: pip install torch transformers pandas pyarrow")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CommonsenseQAEvaluator:
    """Evaluator for CommonsenseQA reasoning tasks."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = Path(model_path)
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        
    def load_model(self) -> bool:
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            logger.info("✓ Tokenizer loaded successfully")
            
            # Load model
            logger.info("Loading model...")
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            # Try to use Flash Attention if available
            try:
                import flash_attn
                if self.device == "cuda":
                    logger.info("Using Flash Attention 2...")
                    model_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.info("Flash Attention not available, using default attention")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info("✓ Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def extract_answer(self, text: str, choices: List[str]) -> Optional[str]:
        """Extract the answer choice from model output."""

        # Look for pattern "#### single letter answer" at the end
        match = re.search(r"####\s*([A-E])\b", text)
        if match:
            return match.group(1).strip()

        # Look for single letter answers (A, B, C, D, E)
        match = re.search(r'\b([A-E])\b', text)
        if match:
            return match.group(1)
        
        # Look for the choice text in the response
        text_lower = text.lower()
        for i, choice in enumerate(choices):
            if choice[1].lower() in text_lower:
                return chr(ord('A') + i)  # Convert to letter
        
        return None
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response for a given prompt."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,  # Use greedy decoding for consistency
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return ""

    def batch_generate_response(self, prompts: list[str], max_new_tokens: int = 512) -> list[str]:
        """Generate responses for a batch of prompts."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")

        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,          # Greedy decoding
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # stop sequences
            stop_words = ["####", "\nQ:", "\nHuman:", "\nUser:", "Q:", "Human:", "User:"]

            cleaned_responses = []
            for idx, full_text in enumerate(decoded):
                # Remove the original prompt part
                text = full_text[len(prompts[idx]):].strip()

                # Truncate at the earliest stop word (including '####')
                stop_positions = [
                    text.find(sw) for sw in stop_words if sw in text
                ]
                if stop_positions:
                    cutoff = min(p for p in stop_positions if p >= 0)
                    # Keep up to and slightly past #### if present
                    if "####" in text[cutoff:cutoff + 50]:
                        text = text[:cutoff + 50]
                    else:
                        text = text[:cutoff]

                cleaned_responses.append(text.strip())

            return cleaned_responses

        except Exception as e:
            logger.error(f"❌ Batch generation failed: {e}")
            return ["" for _ in prompts]
    
    def evaluate_single_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single CommonsenseQA problem."""
        try:
            # Extract problem information
            _id = problem_data['id']
            question = problem_data['question']
            choices = list(zip(problem_data["choices"]["label"].tolist(), problem_data["choices"]["text"].tolist()))
            answer_key = problem_data['answerKey']
            question_concept = problem_data['question_concept']
            
            # Create prompt for CommonsenseQA dataset
            choices_text = "\n".join([f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)])
            prompt = f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"
            
            # Generate response
            response = self.generate_response(prompt)
            
            # Extract answer
            predicted_answer = self.extract_answer(response, choices)
            
            # Check if answer is correct
            is_correct = predicted_answer == answer_key if predicted_answer else False
            
            return {
                'id': _id,
                'problem': question,
                'labels': [c[0] for c in choices],
                'choices': [c[1] for c in choices],
                'solution': answer_key,
                'question_concept': question_concept,
                'prompt': prompt,
                'predicted_answer': predicted_answer,
                'response': response,
                'is_correct': is_correct
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate problem: {e}")
            return {
                'id': _id,
                'problem': 'Error',
                'labels': [],
                'choices': [],
                'solution': '',
                'question_concept': '',
                'prompt': '',
                'predicted_answer': None,
                'response': '',
                'is_correct': False,
                'error': str(e)
            }
    
    def evaluate_batch(
        self, 
        problem_data: list[Dict[str, Any]], 
        prompt_builder: PromptBuilder,
        cache: dict,
    ) -> Dict[str, Any]:
        """Evaluate a batch of GSM8K problems."""
        try:
            # Extract problem information
            _id = [p["id"] for p in problem_data]
            problem = [p["question"] for p in problem_data]
            choices = [list(zip(p["choices"]["label"].tolist(), p["choices"]["text"].tolist())) for p in problem_data]
            solution = [p["answerKey"] for p in problem_data]
            question_concept = [p["question_concept"] for p in problem_data]

            # Create prompt for CommonsenseQA dataset
            # prompt = f"Problem: {problem}\n\nSolution:"
            prompt = [prompt_builder.build_prompt(p, choices=c) for p, c in zip(problem, choices)]

            cached_responses = {}
            uncached_prompts = []
            uncached_indices = []

            for idx, p in enumerate(prompt):
                if p in cache:
                    cached_responses[idx] = cache[p]
                else:
                    uncached_prompts.append(p)
                    uncached_indices.append(idx)

            # Generate responses for uncached prompts
            if uncached_prompts:
                new_responses = self.batch_generate_response(uncached_prompts)
                # Store in cache
                for p, r in zip(uncached_prompts, new_responses):
                    cache[p] = r
                # Map back to their indices
                for idx, r in zip(uncached_indices, new_responses):
                    cached_responses[idx] = r

            # Collect responses in order
            response = [cached_responses[i] for i in range(len(prompt))]

            # Extract answer
            predicted_answer = [self.extract_answer(r, c) for r, c in zip(response, choices)]

            results = []
            for idx, pa in enumerate(predicted_answer):
                is_correct = False
                if pa is not None:
                    # Normalize both answers for comparison
                    pred_normalized = pa.strip().lower()
                    sol_normalized = solution[idx].strip().lower()

                    # Check if predicted answer appears in the solution
                    is_correct = (
                        pred_normalized in sol_normalized
                        or sol_normalized in pred_normalized
                    )

                results.append({
                    "id": _id,
                    "problem": problem[idx],
                    "labels": [c[0] for c in choices[idx]],
                    "choices": [c[1] for c in choices[idx]],
                    "solution": solution[idx],
                    "question_concept": question_concept[idx],
                    "prompt": prompt[idx],
                    "predicted_answer": pa,
                    "response": response[idx],
                    "is_correct": is_correct,
                })

            return results

        except Exception as e:
            logger.error(f"❌ Failed to evaluate batch: {e}")
            return [{
                "id": _id,
                "problem": "Error",
                "labels": [],
                "choices": [],
                "solution": "",
                "question_concept": "",
                "prompt": "",
                "predicted_answer": None,
                "response": "",
                "is_correct": False,
                "error": str(e),
            } for p in problem_data]

    def evaluate_dataset(
        self,
        dataset_path: str,
        cache_path: str,
        prompt_template: Optional[str] = None,
        max_samples: Optional[int] = None,
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """Evaluate the model on the CommonsenseQA dataset."""
        try:
            logger.info(f"Loading dataset from: {dataset_path}")
            df = pd.read_parquet(dataset_path)

            logger.info(f"Loading cache from: {cache_path}")
            with open(cache_path, "w+") as f:
                cache = {}
                try:
                    cache = json.load(f)
                except:
                    pass

            if max_samples:
                df = df.head(max_samples)

            results = []
            correct_count = 0

            if not prompt_template:
                prompt_builder = PromptBuilder()
            elif prompt_template == "cot":
                prompt_builder = COTPromptBuilder()
            elif prompt_template == "icl":
                prompt_builder = ICLPromptBuilder(
                    [
                        (
                            str(row["question"]), 
                            list(zip(row["choices"]["label"].tolist(), row["choices"]["text"].tolist())), 
                            str(row["answerKey"])
                        )
                        for _, row in df.iterrows()
                    ]
                )
            else:
                prompt_builder = PromptBuilder()

            num_batches = math.ceil(len(df)/float(batch_size))
            batches = [None]*num_batches # list of lists of problem_data
            for idx, row in df.iterrows():
                if not batches[int(idx / batch_size)]:
                    batches[int(idx / batch_size)] = []
                batches[int(idx / batch_size)].append(row.to_dict())
            logger.info(f"Evaluating on {len(df)} samples in {num_batches} batches of size {batch_size}")
                

            for idx, b in enumerate(batches):

                if b is None:
                    continue

                t_start = time.time()

                result = self.evaluate_batch(b, prompt_builder, cache)
                results += result

                for r in result:
                    if r["is_correct"]:
                        correct_count += 1

                logger.info(f"Processed batch {idx+1}/{len(batches)} in {round(time.time() - t_start, 3)}")

                # Log progress + cache every 10 batches
                if (idx + 1) % 10 == 0:
                    current_accuracy = correct_count / len(results)
                    logger.info(
                        f"Current accuracy: {current_accuracy:.3f} ({correct_count}/{len(results)})"
                    )

                    if cache_path is not None:
                        with open(cache_path, "w") as f:
                            json.dump(cache , f)
                        logger.info(f"Cached batches 1-{idx+1}")

            if cache_path is not None:
                with open(cache_path, "w+") as f:
                    json.dump(cache , f)
                logger.info(f"Cached all responses")


            # Calculate final metrics
            total_problems = len(results)
            accuracy = correct_count / total_problems if total_problems > 0 else 0

            # Count problems with valid answers
            valid_answers = sum(1 for r in results if r["predicted_answer"] is not None)
            answer_rate = valid_answers / total_problems if total_problems > 0 else 0

            evaluation_results = {
                "total_problems": total_problems,
                "correct_answers": correct_count,
                "accuracy": accuracy,
                "valid_answers": valid_answers,
                "answer_rate": answer_rate,
                "results": results,
                "model_path": str(self.model_path),
                "dataset_path": dataset_path,
                "evaluation_time": datetime.now().isoformat(),
                "device": self.device,
            }

            logger.info(f"✓ Evaluation completed!")
            logger.info(
                f"✓ Accuracy: {accuracy:.3f} ({correct_count}/{total_problems})"
            )
            logger.info(
                f"✓ Answer rate: {answer_rate:.3f} ({valid_answers}/{total_problems})"
            )

            return evaluation_results

        except Exception as e:
            logger.error(f"❌ Dataset evaluation failed: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

def save_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    try:
        # Remove the full results list to keep file size manageable
        # Keep only summary statistics
        summary_results = {
            'total_problems': results['total_problems'],
            'correct_answers': results['correct_answers'],
            'accuracy': results['accuracy'],
            'valid_answers': results['valid_answers'],
            'answer_rate': results['answer_rate'],
            'model_path': results['model_path'],
            'dataset_path': results['dataset_path'],
            'evaluation_time': results['evaluation_time'],
            'device': results['device']
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        logger.info(f"✓ Results saved to: {output_path}")
        
        # Also save detailed results to a separate file
        detailed_path = output_path.replace('.json', '_detailed.json')
        with open(detailed_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ Detailed results saved to: {detailed_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate model on CommonsenseQA dataset')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model directory')
    parser.add_argument('--dataset_path', type=str, 
                       default='/shared/data2/jashrp2/query_conditioned_guidelines/data/commonsenseqa/test.parquet',
                       help='Path to the CommonsenseQA dataset (parquet file)')
    parser.add_argument('--output_path', type=str,
                       help='Path to save evaluation results (JSON file)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        choices=[None, "cot", "icl"],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default=None
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CommonsenseQA Baseline Evaluation")
    print("=" * 80)
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model path does not exist: {args.model_path}")
        return False
    
    # Check if dataset path exists
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset path does not exist: {args.dataset_path}")
        return False
    
    # Set default output path if not provided
    if not args.output_path:
        model_name = Path(args.model_path).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f"commonsenseqa_baseline_{model_name}_{timestamp}.json"
    
    # Initialize evaluator
    evaluator = CommonsenseQAEvaluator(args.model_path, args.device)
    
    # Load model
    print(f"\n1. Loading Model...")
    if not evaluator.load_model():
        print("❌ Failed to load model")
        return False
    
    # Run evaluation
    print(f"\n2. Running Evaluation...")
    results = evaluator.evaluate_dataset(
        args.dataset_path, 
        args.cache_path, 
        prompt_template=args.prompt_template, 
        max_samples=args.max_samples,
        batch_size=args.batch_size
    ) 
    
    if 'error' in results:
        print(f"❌ Evaluation failed: {results['error']}")
        return False
    
    # Save results
    print(f"\n3. Saving Results...")
    save_results(results, args.output_path)
    
    # Print summary
    print(f"\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Model: {results['model_path']}")
    print(f"Dataset: {results['dataset_path']}")
    print(f"Total Problems: {results['total_problems']}")
    print(f"Correct Answers: {results['correct_answers']}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Answer Rate: {results['answer_rate']:.3f}")
    print(f"Device: {results['device']}")
    print(f"Results saved to: {args.output_path}")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
