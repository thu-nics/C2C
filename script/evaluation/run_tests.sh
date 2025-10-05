#!/bin/bash

# 测试配置
MODELS=("Qwen3-0.6B" "Qwen3-1.7B" "Rosetta")
METHODS=("zero_shot" "few_shot")
ANSWER_METHODS=("logits" )
GPU_ID=1  # 设置使用的GPU ID
MAX_LENGTH=32768
BATCH_SIZE=1

# 遍历所有配置组合
for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
        for answer_method in "${ANSWER_METHODS[@]}"; do
            
            # 处理zero_shot情况（只运行ntrain=0）
            if [ "$method" == "zero_shot" ]; then
                ntrain=0
                
                echo "=============================================="
                echo "Running: model=$model, method=$method, answer_method=$answer_method, ntrain=$ntrain"
                echo "=============================================="
                
                python evaluator.py \
                    --model_name $model \
                    --method $method \
                    --answer_method $answer_method \
                    --gpu_id $GPU_ID \
                    --ntrain $ntrain \
                    --max_length $MAX_LENGTH \
                    --batch_size $BATCH_SIZE
            
            # 处理few_shot情况（运行ntrain=1到10）
            else
                for ntrain in {1..10}; do
                    echo "=============================================="
                    echo "Running: model=$model, method=$method, answer_method=$answer_method, ntrain=$ntrain"
                    echo "=============================================="
                    
                    python evaluator.py \
                        --model_name $model \
                        --method $method \
                        --answer_method $answer_method \
                        --gpu_id $GPU_ID \
                        --ntrain $ntrain \
                        --max_length $MAX_LENGTH \
                        --batch_size $BATCH_SIZE
                done
            fi
            
        done
    done
done

echo "All tests completed!"