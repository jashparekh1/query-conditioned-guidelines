# 启服务（如尚未启动）
# CUDA_VISIBLE_DEVICES=6,7 PORT=1225 MODEL_PATH=merge_models/546-nlp-gsm8k-grpo-qwen-2.5-3b-instruct MODEL_NAME=base-llm \
# bash /shared/data2/jiashuo5/verl/serve_base_llm.sh

# sleep 180
# 跑评测
PYTHONUNBUFFERED=1 python3 -m eval.gsm8k_base_eval \
  --dataset_path "/shared/data2/jiashuo5/verl/data/gsm8k" \
  --split test \
  --base_url http://127.0.0.1:1225/v1 \
  --model_name base-llm \
  --output_path gsm8k_base_eval_results.jsonl