#!/usr/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,3

# training 
cd /mnt/yht/code/HeltonPretrain

# fcos_pafpn_dinov3sta_coco
PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port=29558 \
    detection/tools/train.py \
    --config /mnt/yht/code/HeltonPretrain/detection/configs/fcos_pafpn_dinov3sta_coco_ddp.py
