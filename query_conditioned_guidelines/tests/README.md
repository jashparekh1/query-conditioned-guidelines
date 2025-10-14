# Qwen2.5-3B-Instruct Inference Test

This directory contains a test script to verify that the Qwen2.5-3B-Instruct model can run inference properly.

## Files

- `test_qwen_inference.py` - Main test script for model inference
- `requirements.txt` - Required Python dependencies

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the inference test:
   ```bash
   python tests/test_qwen_inference.py
   ```

   Or make it executable and run directly:
   ```bash
   chmod +x tests/test_qwen_inference.py
   ./tests/test_qwen_inference.py
   ```

## What the test does

The test script performs the following checks:

1. **Model Loading**: Loads the Qwen2.5-3B-Instruct model and tokenizer
2. **Model Information**: Displays model parameters, size, and configuration
3. **Basic Inference**: Tests simple text generation with various prompts
4. **Chat Format**: Tests the model's chat/instruct capabilities
5. **Multiple Prompts**: Tests inference with different types of prompts

## Expected Output

The script will show:
- ✓ for successful operations
- ❌ for failures
- Model information (parameters, size, device)
- Generated responses for each test

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.40+
- CUDA (optional, for GPU acceleration)
- Sufficient RAM/VRAM for the 3B parameter model

## Troubleshooting

- If you get CUDA out of memory errors, the script will automatically fall back to CPU
- For memory-constrained environments, consider using quantization (modify the script to set `use_quantization=True`)
- Ensure the model path is correct and all model files are present
