# Defaults (override by exporting env vars before calling this script)
BASE_CONFIG=${BASE_CONFIG:-recipe/train_recipe/ablation_base.json}
BASE_EVAL_CONFIG=${BASE_EVAL_CONFIG:-recipe/eval_recipe/ablation_base.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-local/checkpoints/ablation_study_general}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-local/ablation_results_general}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
ABLATION_LEVELS=${ABLATION_LEVELS:-0,1,2,3,4}
MASTER_PORT=${MASTER_PORT:-29504}

# GPU visibility
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# Forward any additional flags directly to the Python script
python script/ablation/ablation_study.py \
  --base_config "$BASE_CONFIG" \
  --base_eval_config "$BASE_EVAL_CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --eval_output_dir "$EVAL_OUTPUT_DIR" \
  --gpu_ids "$GPU_IDS" \
  --ablation_levels "$ABLATION_LEVELS" \
  --master_port "$MASTER_PORT" \
  "$@"
