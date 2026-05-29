export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 --master_port=29513 script/train/SFT_train_v3.py \
    --config recipe/train_recipe/C2C_v3_hidden_0.6+0.5_2gpu.json
