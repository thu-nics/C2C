#!/bin/bash
# SFT Training on UltraChat Dataset
#
# Usage:
#   bash script/train/run_sft_ultrachat.sh

set -e

# Configuration
MODEL_NAME="Qwen/Qwen3-1.7B"
OUTPUT_DIR="./checkpoints/sft_ultrachat"
NUM_GPUS=8

# Training hyperparameters
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-5
MAX_LENGTH=4096
MAX_STEPS=1000
WARMUP_STEPS=100
SAVE_STEPS=500
LOG_STEPS=10

# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS
# = 2 * 8 * 8 = 128

WANDB_RUN_NAME="sft-qwen3-1.7b-ultrachat"

echo "=========================================="
echo "SFT Training on UltraChat"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))"
echo "Max steps: ${MAX_STEPS}"
echo "=========================================="

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rosetta

# Change to project root
cd /share/futianyu/cloud/repo/release/C2C_release

# Run training
accelerate launch --num_processes ${NUM_GPUS} \
    script/train/sft_ultrachat.py \
    --model_name ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --max_length ${MAX_LENGTH} \
    --max_steps ${MAX_STEPS} \
    --warmup_steps ${WARMUP_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --log_steps ${LOG_STEPS} \
    --wandb_run_name ${WANDB_RUN_NAME}

echo "=========================================="
echo "Training Complete"
echo "=========================================="

# Run evaluation
echo ""
echo "=========================================="
echo "Running Evaluation"
echo "=========================================="

python script/eval/eval_chat.py \
    --model_path ${OUTPUT_DIR}/final

echo "=========================================="
echo "All Done!"
echo "=========================================="

