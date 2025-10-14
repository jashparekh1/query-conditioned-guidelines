#!/bin/bash
# GSM8K Baseline Evaluation Runner
# Simple wrapper script to run GSM8K evaluation with common configurations

MODEL_PATH="/shared/data2/jashrp2/query_conditioned_guidelines/models/frozen/Qwen2.5-3B-Instruct"
DATASET_PATH="/shared/data2/jashrp2/query_conditioned_guidelines/data/gsm8k/test.parquet"
SCRIPT_PATH="/shared/data2/jashrp2/query_conditioned_guidelines/tests/evaluate_gsm8k_baseline.py"

# Default values
MAX_SAMPLES=""
OUTPUT_PATH=""
DEVICE="auto"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-samples)
            MAX_SAMPLES="--max_samples $2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="--output_path $2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --max-samples N    Evaluate only first N samples (for testing)"
            echo "  --output PATH      Output file path for results"
            echo "  --device DEVICE    Device to use (auto, cuda, cpu)"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Full evaluation"
            echo "  $0 --max-samples 10                  # Test with 10 samples"
            echo "  $0 --device cpu                      # Force CPU usage"
            echo "  $0 --output my_results.json          # Custom output file"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "GSM8K Baseline Evaluation Runner"
echo "================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Device: $DEVICE"
echo ""

# Run the evaluation
python "$SCRIPT_PATH" \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --device "$DEVICE" \
    $MAX_SAMPLES \
    $OUTPUT_PATH

echo ""
echo "Evaluation completed!"
