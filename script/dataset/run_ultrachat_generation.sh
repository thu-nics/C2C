#!/bin/bash

# UltraChat Dataset Generation Script
#
# This script generates responses for the UltraChat dataset using SGLang.
# Make sure the SGLang server is running before executing this script!
#
# Usage:
#   # First, launch the server in another terminal:
#   bash script/dataset/launch_ultrachat_server.sh
#   
#   # Then run this script:
#   bash script/dataset/run_ultrachat_generation.sh
#
# Options (set as environment variables):
#   MODEL_PATH: Path to the model (default: /share/public/public_models/Qwen3-8B)
#   API_URL: SGLang server URL (default: http://localhost:30000/v1)
#   OUTPUT_DIR: Output directory (default: local/ultrachat_qwen3_8b_output)
#   NUM_ITEMS: Number of items to process (default: all)
#   MAX_CONCURRENT: Max concurrent requests (default: 256)

set -e

# Configuration with defaults
MODEL_PATH=${MODEL_PATH:-"/share/public/public_models/Qwen3-8B"}
API_URL=${API_URL:-"http://localhost:30000/v1"}
OUTPUT_DIR=${OUTPUT_DIR:-"local/ultrachat_qwen3_8b_output"}
NUM_ITEMS=${NUM_ITEMS:-""}  # Empty = all items
MAX_CONCURRENT=${MAX_CONCURRENT:-256}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}

echo "========================================"
echo "🚀 Starting UltraChat Dataset Generation"
echo "========================================"
echo "Model: ${MODEL_PATH}"
echo "API URL: ${API_URL}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Max Concurrent Requests: ${MAX_CONCURRENT}"
echo "Max New Tokens: ${MAX_NEW_TOKENS}"
echo "Temperature: ${TEMPERATURE}"
echo "Top-p: ${TOP_P}"
echo "Top-k: ${TOP_K}"
if [ -n "${NUM_ITEMS}" ]; then
    echo "Number of items: ${NUM_ITEMS}"
else
    echo "Number of items: ALL"
fi
echo "========================================"

# Build command
CMD="python script/dataset/create_ultrachat.py \
    --model_path \"${MODEL_PATH}\" \
    --api_url \"${API_URL}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --max_concurrent_requests ${MAX_CONCURRENT} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K} \
    --save_every 1000 \
    --request_timeout 600"

# Add optional num_items
if [ -n "${NUM_ITEMS}" ]; then
    CMD="${CMD} --num_items ${NUM_ITEMS}"
fi

# Execute
echo ""
echo "Executing: ${CMD}"
echo ""
eval ${CMD}

echo ""
echo "✅ Generation completed!"
echo "Results saved to: ${OUTPUT_DIR}"

