#!/usr/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,3

# training 
cd /mnt/yht/code/HeltonPretrain


# mlpnet
# /mnt/yht/env/yht_pretrain/bin/python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=29558 \
#     tools/train.py \
#     --config /mnt/yht/code/HeltonPretrain/configs/mlpnet_ddp.py

# protonet
# /mnt/yht/env/yht_pretrain/bin/python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=29558 \
#     tools/train.py \
#     --config /mnt/yht/code/HeltonPretrain/configs/protonet_ddp.py


# protonet_dinov3_vits
# /mnt/yht/env/yht_pretrain/bin/python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=29558 \
#     tools/train.py \
#     --config /mnt/yht/code/HeltonPretrain/configs/protonet_dinov3vits_ddp.py


# mlpnet_dinov3vits_vithead_ddp
# /mnt/yht/env/yht_pretrain/bin/python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=29558 \
#     tools/train.py \
#     --config /mnt/yht/code/HeltonPretrain/configs/mlpnet_dinov3vits_vithead_ddp.py

# mlp_distill_net
# /mnt/yht/env/yht_pretrain/bin/python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port=29558 \
#     tools/train.py \
#     --config /mnt/yht/code/HeltonPretrain/configs/mlpnet_dinov3_distill_ddp.py

# mlp_net_multi_tasks
/mnt/yht/env/yht_pretrain/bin/python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_port=29558 \
    tools/train.py \
    --config /mnt/yht/code/HeltonPretrain/configs/mlpnet_multi_tasks_distill_clip_ddp.py