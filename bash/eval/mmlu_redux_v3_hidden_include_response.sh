#!/usr/bin/env bash
set -euo pipefail

python script/evaluation/unified_evaluator_v3.py \
    --config recipe/eval_recipe/mmlu_redux_v3_hidden_include_response.yaml
