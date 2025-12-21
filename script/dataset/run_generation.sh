#!/bin/bash

# Simple script to run OpenHermes dataset generation
# Make sure the server is running first!

echo "🚀 Starting dataset generation..."

python script/dataset/create/create_gsm8k.py \
    --model_path "Qwen/Qwen3-32B" \
    --api_url "http://localhost:30000/v1" \
    --dataset_path "openai/gsm8k" \
    --output_dir "local/teacher_datasets/gsm8k_qwen3_32b_output_test" \
    --max_concurrent_requests 256 \
    --max_new_tokens 1024 \
    --split test \
    --temperature 0 \
    --top_p 0.95 \
    --top_k 20 \
    --min_p 0.0 \
    --request_timeout 6000 \
    --save_every 100 \

echo "✅ Generation completed!"
