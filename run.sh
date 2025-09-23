#!/usr/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3

# training 
# /home/kpn/anaconda3/envs/yht_mmdet3.x run -n yht_mmdet3.x
cd /mnt/yht/code/HeltonPretrain



/home/kpn/anaconda3/envs/yht_mmdet3.x/bin/python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=29558 \
    runner.py \
    --config /mnt/yht/code/HeltonPretrain/configs/fcnet.py