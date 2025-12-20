export CUDA_VISIBLE_DEVICES=7
torchrun --nproc_per_node=1 --master_port=29504 script/train/SFT_train.py \
    --config recipe/train_recipe/include_response.json
    # --resume_from_checkpoint local/checkpoints/qwen3_0.6b+qwen3_32b_gsm8k_include_response/checkpoint-100