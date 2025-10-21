TRAIN_MODEL_NAME=checkpoints/546-nlp-gsm8k-grpo/gsm8k-grpo-qwen-2.5-3b-instruct/global_step_580/actor
TARGET_DIR=merge_models/546-nlp-gsm8k-grpo-qwen-2.5-3b-instruct

python scripts/merge_model.py merge --backend fsdp \
    --hf_model_path Qwen/Qwen2.5-3B-Instruct \
    --local_dir "$TRAIN_MODEL_NAME" \
    --target_dir "$TARGET_DIR"