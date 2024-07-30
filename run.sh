#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3



# training
/usr/local/anaconda3/bin/conda run -n hd
cd /data/cs/huawei/code/HeltonPretrain

/home/cs/.conda/envs/hd/bin/python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=29558\
    runner.py \
    --config ./configs/clipdistillclsnet.py