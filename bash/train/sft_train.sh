export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=1 --master_port=29504 script/train/SFT_train.py \
    --config recipe/train_recipe/C2C_0.6+0.5.json \