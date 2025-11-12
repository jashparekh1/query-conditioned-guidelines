#!/usr/bin/env python3
"""
Test script for Qwen2.5-3B-Instruct model inference.
This script tests basic inference capabilities of the downloaded model.
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
import dotenv

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

dotenv.load_dotenv()

try:
    import torch
    import transformers
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        GenerationConfig,
        BitsAndBytesConfig
    )
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ Transformers version: {transformers.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device count: {torch.cuda.device_count()}")
        print(f"✓ Current CUDA device: {torch.cuda.current_device()}")
    
    # Check for Flash Attention availability
    try:
        import flash_attn
        print(f"✓ Flash Attention available: {flash_attn.__version__}")
        FLASH_ATTENTION_AVAILABLE = True
    except ImportError:
        print("⚠ Flash Attention not available - may see sliding window attention warnings")
        FLASH_ATTENTION_AVAILABLE = False
        
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("Please install: pip install torch transformers accelerate bitsandbytes")
    print("For better attention support: pip install flash-attn --no-build-isolation")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QwenInferenceTester:
    """Test class for Qwen2.5-3B-Instruct model inference."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self, use_quantization: bool = False) -> bool:
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
            
            # Configure model loading
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            # Use Flash Attention if available to avoid sliding window attention warnings
            if FLASH_ATTENTION_AVAILABLE and self.device == "cuda":
                logger.info("Using Flash Attention 2 to avoid sliding window attention warnings...")
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            # Add quantization if requested and CUDA is available
            if use_quantization and self.device == "cuda":
                logger.info("Using 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
            
            # Load model
            logger.info("Loading model...")
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
    
    def test_basic_inference(self, prompt: str = "Hello, how are you?") -> Dict[str, Any]:
        """Test basic text generation."""
        if not self.model or not self.tokenizer:
            return {"success": False, "error": "Model not loaded"}
        
        try:
            logger.info(f"Testing basic inference with prompt: '{prompt}'")
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            logger.info(f"✓ Generated response: '{generated_text}'")
            
            return {
                "success": True,
                "prompt": prompt,
                "response": generated_text,
                "full_response": response,
                "input_length": len(inputs.input_ids[0]),
                "output_length": len(outputs[0])
            }
            
        except Exception as e:
            logger.error(f"❌ Basic inference failed: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def test_chat_format(self, messages: list = None) -> Dict[str, Any]:
        """Test chat/instruct format."""
        if not self.model or not self.tokenizer:
            return {"success": False, "error": "Model not loaded"}
        
        if messages is None:
            messages = [
                {"role": "user", "content": "What is the capital of France?"}
            ]
        
        try:
            logger.info("Testing chat format...")
            
            # Apply chat template
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            logger.info(f"Chat prompt: {chat_prompt}")
            
            # Tokenize and generate
            inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(chat_prompt):].strip()
            
            logger.info(f"✓ Chat response: '{generated_text}'")
            
            return {
                "success": True,
                "messages": messages,
                "chat_prompt": chat_prompt,
                "response": generated_text,
                "full_response": response
            }
            
        except Exception as e:
            logger.error(f"❌ Chat format test failed: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def test_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        try:
            info = {
                "model_path": str(self.model_path),
                "device": self.device,
                "model_loaded": self.model is not None,
                "tokenizer_loaded": self.tokenizer is not None,
            }
            
            if self.model:
                info.update({
                    "model_config": self.model.config.to_dict(),
                    "model_parameters": sum(p.numel() for p in self.model.parameters()),
                    "model_size_gb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**3),
                })
            
            if self.tokenizer:
                info.update({
                    "vocab_size": self.tokenizer.vocab_size,
                    "model_max_length": getattr(self.tokenizer, 'model_max_length', 'Unknown'),
                })
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Failed to get model info: {e}")
            return {"success": False, "error": str(e)}

def main():
    """Main test function."""
    # model_path = "/shared/data2/jashrp2/query_conditioned_guidelines/models/frozen/Qwen2.5-3B-Instruct"
    model_path = os.getenv("MODEL_DIR") + "frozen/Qwen2.5-3B-Instruct"
    
    print("=" * 60)
    print("Qwen2.5-3B-Instruct Inference Test")
    print("=" * 60)
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    # Initialize tester
    tester = QwenInferenceTester(model_path)
    
    # Test 1: Load model
    print("\n1. Loading Model...")
    success = tester.load_model(use_quantization=False)  # Try without quantization first
    if not success:
        print("❌ Failed to load model")
        return False
    
    # Test 2: Model information
    print("\n2. Model Information...")
    model_info = tester.test_model_info()
    if model_info.get("success", True):  # Model info doesn't have success field
        print(f"✓ Model parameters: {model_info.get('model_parameters', 'Unknown'):,}")
        print(f"✓ Model size: {model_info.get('model_size_gb', 'Unknown'):.2f} GB")
        print(f"✓ Vocab size: {model_info.get('vocab_size', 'Unknown'):,}")
        print(f"✓ Device: {model_info.get('device', 'Unknown')}")
    
    # Test 3: Basic inference
    print("\n3. Basic Inference Test...")
    basic_result = tester.test_basic_inference("The weather today is")
    if basic_result["success"]:
        print(f"✓ Prompt: {basic_result['prompt']}")
        print(f"✓ Response: {basic_result['response']}")
    else:
        print(f"❌ Basic inference failed: {basic_result['error']}")
    
    # Test 4: Chat format
    print("\n4. Chat Format Test...")
    chat_result = tester.test_chat_format([
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ])
    if chat_result["success"]:
        print(f"✓ Chat response: {chat_result['response']}")
    else:
        print(f"❌ Chat format test failed: {chat_result['error']}")
    
    # Test 5: Multiple prompts
    print("\n5. Multiple Prompts Test...")
    test_prompts = [
        "Write a haiku about programming:",
        "What are the benefits of renewable energy?",
        "Solve this math problem: 15 * 23 ="
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n5.{i} Testing: {prompt}")
        result = tester.test_basic_inference(prompt)
        if result["success"]:
            print(f"✓ Response: {result['response']}")
        else:
            print(f"❌ Failed: {result['error']}")
    
    print("\n" + "=" * 60)
    print("Inference test completed!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
