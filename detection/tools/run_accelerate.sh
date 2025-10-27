#!/usr/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,3
# export CUDA_VISIBLE_DEVICES=1,2

# training 
cd /mnt/yht/code/HeltonPretrain


# fcos_voc
# /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file configs/accelerate_yamls/accelerate_ddp.yaml \
#     tools/train_accelerate.py \
#     --config /mnt/yht/code/HeltonPretrain/detection/configs/fcos_VOC_ddp.py

# fcos_coco
# /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file configs/accelerate_yamls/accelerate_ddp.yaml \
#     tools/train_accelerate.py \
#     --config /mnt/yht/code/HeltonPretrain/detection/configs/fcos_coco_ddp.py

# fcos_pafpn_coco
# /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file configs/accelerate_yamls/accelerate_ddp.yaml \
#     tools/train_accelerate.py \
#     --config /mnt/yht/code/HeltonPretrain/detection/configs/fcos_pafpn_coco_ddp.py

# fcos_pafpn_dinov3sta_coco
/mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file configs/accelerate_yamls/accelerate_ddp.yaml \
    tools/train_accelerate.py \
    --config /mnt/yht/code/HeltonPretrain/detection/configs/fcos_pafpn_dinov3sta_coco_ddp.py

# fcos_c2fpafpn_dinov3sta_coco
# /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file configs/accelerate_yamls/accelerate_ddp.yaml \
#     tools/train_accelerate.py \
#     --config /mnt/yht/code/HeltonPretrain/detection/configs/fcos_c2fpafpn_dinov3sta_coco_ddp.py