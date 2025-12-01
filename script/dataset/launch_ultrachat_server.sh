#!/bin/bash

# SGLang Server Launch Script for UltraChat Generation
# 
# This script launches an SGLang server with Qwen3-8B for fast batch generation.
# The server uses data parallelism for optimal throughput on multi-GPU systems.
#
# Usage:
#   bash script/dataset/launch_ultrachat_server.sh
#
# Options (set as environment variables):
#   MODEL_PATH: Path to the model (default: /share/public/public_models/Qwen3-8B)
#   PORT: Server port (default: 30000)
#   TP_SIZE: Tensor parallel size (default: 1)
#   DP_SIZE: Data parallel size (default: 8, uses all available GPUs)
#   CUDA_DEVICES: GPU devices to use (default: 0,1,2,3,4,5,6,7)

set -e

# Configuration with defaults
MODEL_PATH=${MODEL_PATH:-"/share/public/public_models/Qwen3-1.7B"}
PORT=${PORT:-30000}
TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-8}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1,2,3,4,5,6,7"}

echo "========================================"
echo "🚀 Launching SGLang Server for UltraChat"
echo "========================================"
echo "Model: ${MODEL_PATH}"
echo "Port: ${PORT}"
echo "TP Size: ${TP_SIZE}"
echo "DP Size: ${DP_SIZE}"
echo "CUDA Devices: ${CUDA_DEVICES}"
echo "========================================"

# Get the script directory for the chat template path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAT_TEMPLATE="${SCRIPT_DIR}/qwen3_nonthinking.jinja"

# Check if chat template exists
if [ -f "${CHAT_TEMPLATE}" ]; then
    echo "Using chat template: ${CHAT_TEMPLATE}"
    CHAT_TEMPLATE_ARG="--chat-template ${CHAT_TEMPLATE}"
else
    echo "Warning: Chat template not found at ${CHAT_TEMPLATE}"
    echo "Using default model chat template (enable_thinking=False via API)"
    CHAT_TEMPLATE_ARG=""
fi

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}

# Launch SGLang server
# Key parameters:
# - tp-size: Tensor parallelism (1 for 8B model is fine)
# - dp-size: Data parallelism (8 for 8 GPUs = 8 replicas for parallel requests)
# - mem-fraction-static: Memory allocation (0.9 = 90% for KV cache)
# - dtype: Use bfloat16 for Qwen3 models
python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port ${PORT} \
    --tp-size ${TP_SIZE} \
    --dp-size ${DP_SIZE} \
    --mem-fraction-static 0.9 \
    --dtype bfloat16 \
    --log-level warning \
    ${CHAT_TEMPLATE_ARG}

echo "Server stopped."

