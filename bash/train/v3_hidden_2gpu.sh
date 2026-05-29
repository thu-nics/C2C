#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MASTER_PORT="${MASTER_PORT:-29513}"

torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" script/train/SFT_train_v3.py \
    --config recipe/train_recipe/C2C_v3_hidden_0.6+0.5_2gpu.json
