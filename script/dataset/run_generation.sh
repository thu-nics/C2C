#!/bin/bash

# Simple script to run OpenHermes dataset generation
# Make sure the server is running first!

echo "🚀 Starting dataset generation..."

python script/dataset/create/create_mmlu.py \
    --model_path "Qwen/Qwen3-4B" \
    --api_url "http://localhost:30000/v1" \
    --dataset_path "cais/mmlu" \
    --output_dir "local/teacher_datasets/mmlu_4b_output_150_words" \
    --max_concurrent_requests 256 \
    --max_new_tokens 512 \
    --split auxiliary_train \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --min_p 0.0 \
    --request_timeout 6000 \
    --save_every 100 \
    --sample_every_n 6 \

echo "✅ Generation completed!"
