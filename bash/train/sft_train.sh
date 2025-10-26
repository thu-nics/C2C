export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 --master_port=29501 script/train/SFT_train.py \
    --config recipe/train_recipe/C2C_0.6+0.5.json \