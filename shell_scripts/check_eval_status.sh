#!/bin/bash
# Quick status check for running evaluation jobs

echo "=========================================="
echo "Evaluation Job Status"
echo "=========================================="
echo ""

# Check running jobs
echo "Running Jobs:"
squeue -u $USER -o "%.10i %.12P %.20j %.8u %.8T %.10M %.6D %R" | head -5
echo ""

# Check output files and progress
echo "Progress by Dataset:"
echo ""

for dataset in gsm8k math commonsenseqa strategyqa; do
    dir="${dataset}_two_stage_outputs"
    if [ -d "$dir" ]; then
        echo "--- $dataset ---"
        
        # Check guidelines file
        if [ -f "$dir/${dataset}_"*"_guidelines.jsonl" ]; then
            guideline_file=$(ls "$dir/${dataset}_"*"_guidelines.jsonl" 2>/dev/null | head -1)
            if [ -n "$guideline_file" ]; then
                guideline_lines=$(wc -l < "$guideline_file" 2>/dev/null || echo 0)
                echo "  Guidelines: $guideline_lines lines"
            fi
        fi
        
        # Check predictions file
        if [ -f "$dir/${dataset}_"*"_predictions.jsonl" ]; then
            pred_file=$(ls "$dir/${dataset}_"*"_predictions.jsonl" 2>/dev/null | head -1)
            if [ -n "$pred_file" ]; then
                pred_lines=$(wc -l < "$pred_file" 2>/dev/null || echo 0)
                echo "  Predictions: $pred_lines lines"
            fi
        fi
        
        # Check summary file (indicates completion)
        if [ -f "$dir/${dataset}_"*"_summary.txt" ]; then
            echo "  Status: ✓ COMPLETED"
            if [ -f "$dir/${dataset}_"*"_summary.txt" ]; then
                summary_file=$(ls "$dir/${dataset}_"*"_summary.txt" 2>/dev/null | head -1)
                echo "  Results:"
                tail -3 "$summary_file" 2>/dev/null | sed 's/^/    /'
            fi
        else
            echo "  Status: ⏳ IN PROGRESS"
        fi
        echo ""
    else
        echo "--- $dataset ---"
        echo "  Status: ⏳ Not started yet"
        echo ""
    fi
done

echo "=========================================="
echo "Recent Log Activity (last 3 lines):"
echo "=========================================="
for log in logs/eval_*.out; do
    if [ -f "$log" ]; then
        echo ""
        echo "--- $(basename $log) ---"
        tail -3 "$log" 2>/dev/null | sed 's/^/  /'
    fi
done

