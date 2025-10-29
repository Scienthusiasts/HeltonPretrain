#!/usr/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,3

# training 
cd /mnt/yht/code/HeltonPretrain



# ddpm_unet_FlickrBreeds
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port=29558 \
#     generation/tools/train.py \
#     --config generation/configs/ddpm_unet_ddp.py

# ddpm_unet_DIOR
PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port=29558 \
    generation/tools/train.py \
    --config generation/configs/ddpm_unet_DIOR_ddp.py