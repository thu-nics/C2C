#!/bin/bash

# Simple script to launch SGLang server for OpenHermes dataset generation

echo "🚀 Launching SGLang server..."

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m sglang.launch_server \
    --model-path /share/public/public_models/Qwen3-4B \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 1 \
    --dp-size 8 \
    --mem-fraction-static 0.9 \
    --dtype bfloat16 \
    --log-level warning \
    --chat-template script/dataset/create/qwen3_nonthinking.jinja

echo "Server stopped."
