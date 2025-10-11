#!/usr/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,3
export CUDA_VISIBLE_DEVICES=1,2

# training 
cd /mnt/yht/code/HeltonPretrain


# ddpm_unet
/mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file configs/accelerate_yamls/accelerate_ddp.yaml \
    tools/train_accelerate.py \
    --config /mnt/yht/code/HeltonPretrain/generation/configs/ddpm_unet_ddp.py

