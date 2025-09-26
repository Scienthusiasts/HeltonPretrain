#!/usr/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3

# training 
cd /mnt/yht/code/HeltonPretrain


# fcnet
# /mnt/yht/env/yht_pretrain/bin/python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=29558 \
#     tools/train.py \
#     --config /mnt/yht/code/HeltonPretrain/configs/fcnet_ddp.py

# protonet
# /mnt/yht/env/yht_pretrain/bin/python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=29558 \
#     tools/train.py \
#     --config /mnt/yht/code/HeltonPretrain/configs/protonet_ddp.py


# protonet_dinov3_vits
/mnt/yht/env/yht_pretrain/bin/python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=29558 \
    tools/train.py \
    --config /mnt/yht/code/HeltonPretrain/configs/protonet_dinov3vits_ddp.py