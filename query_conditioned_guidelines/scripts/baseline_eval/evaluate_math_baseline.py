#!/usr/bin/env python3
"""
MATH Baseline Evaluation Script
Evaluates a model's performance on the MATH dataset and calculates accuracy metrics.
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

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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

class MATHEvaluator:
    """Evaluator for MATH mathematical reasoning tasks."""
    
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
            
            logger.info("Using default attention implementation")
            
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
    
    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the final answer from model output."""
        # Look for pattern "\boxed{answer}" which is common in MATH dataset
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1)
        
        # Look for pattern "#### answer" at the end
        match = re.search(r'####\s*([^\n]+)', text)
        if match:
            return match.group(1).strip()
        
        # Fallback: look for the last line that might contain an answer
        lines = text.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if last_line and not last_line.startswith('Step') and not last_line.startswith('Let'):
                return last_line
        
        return None
    
    def generate_response(self, prompt: str, max_new_tokens: int = 1024) -> str:
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
    
    def evaluate_single_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single MATH problem."""
        try:
            # Extract problem information
            problem = problem_data['problem']
            solution = problem_data['solution']
            level = problem_data['level']
            problem_type = problem_data['type']
            
            # Create prompt for MATH dataset
            prompt = f"Problem: {problem}\n\nSolution:"
            
            # Generate response
            response = self.generate_response(prompt)
            
            # Extract answer
            predicted_answer = self.extract_answer(response)
            
            # For MATH dataset, we'll do a simple string comparison
            # In practice, you might want more sophisticated answer matching
            is_correct = False
            if predicted_answer is not None:
                # Normalize both answers for comparison
                pred_normalized = predicted_answer.strip().lower()
                sol_normalized = solution.strip().lower()
                
                # Check if predicted answer appears in the solution
                is_correct = pred_normalized in sol_normalized or sol_normalized in pred_normalized
            
            return {
                'problem': problem,
                'solution': solution,
                'level': level,
                'type': problem_type,
                'prompt': prompt,
                'predicted_answer': predicted_answer,
                'response': response,
                'is_correct': is_correct
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate problem: {e}")
            return {
                'problem': 'Error',
                'solution': '',
                'level': '',
                'type': '',
                'prompt': '',
                'predicted_answer': None,
                'response': '',
                'is_correct': False,
                'error': str(e)
            }
    
    def evaluate_dataset(self, dataset_path: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate the model on the MATH dataset."""
        try:
            logger.info(f"Loading dataset from: {dataset_path}")
            df = pd.read_parquet(dataset_path)
            
            if max_samples:
                df = df.head(max_samples)
                logger.info(f"Evaluating on {max_samples} samples")
            else:
                logger.info(f"Evaluating on {len(df)} samples")
            
            results = []
            correct_count = 0
            
            for idx, row in df.iterrows():
                logger.info(f"Processing problem {idx + 1}/{len(df)}")
                
                result = self.evaluate_single_problem(row.to_dict())
                results.append(result)
                
                if result['is_correct']:
                    correct_count += 1
                
                # Log progress every 10 problems
                if (idx + 1) % 10 == 0:
                    current_accuracy = correct_count / (idx + 1)
                    logger.info(f"Current accuracy: {current_accuracy:.3f} ({correct_count}/{idx + 1})")
            
            # Calculate final metrics
            total_problems = len(results)
            accuracy = correct_count / total_problems if total_problems > 0 else 0
            
            # Count problems with valid answers
            valid_answers = sum(1 for r in results if r['predicted_answer'] is not None)
            answer_rate = valid_answers / total_problems if total_problems > 0 else 0
            
            evaluation_results = {
                'total_problems': total_problems,
                'correct_answers': correct_count,
                'accuracy': accuracy,
                'valid_answers': valid_answers,
                'answer_rate': answer_rate,
                'results': results,
                'model_path': str(self.model_path),
                'dataset_path': dataset_path,
                'evaluation_time': datetime.now().isoformat(),
                'device': self.device
            }
            
            logger.info(f"✓ Evaluation completed!")
            logger.info(f"✓ Accuracy: {accuracy:.3f} ({correct_count}/{total_problems})")
            logger.info(f"✓ Answer rate: {answer_rate:.3f} ({valid_answers}/{total_problems})")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Dataset evaluation failed: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}

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
    parser = argparse.ArgumentParser(description='Evaluate model on MATH dataset')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model directory')
    parser.add_argument('--dataset_path', type=str, 
                       default='/shared/data2/jashrp2/query_conditioned_guidelines/data/math/test.parquet',
                       help='Path to the MATH dataset (parquet file)')
    parser.add_argument('--output_path', type=str,
                       help='Path to save evaluation results (JSON file)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MATH Baseline Evaluation")
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
        args.output_path = f"math_baseline_{model_name}_{timestamp}.json"
    
    # Initialize evaluator
    evaluator = MATHEvaluator(args.model_path, args.device)
    
    # Load model
    print(f"\n1. Loading Model...")
    if not evaluator.load_model():
        print("❌ Failed to load model")
        return False
    
    # Run evaluation
    print(f"\n2. Running Evaluation...")
    results = evaluator.evaluate_dataset(args.dataset_path, args.max_samples)
    
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
