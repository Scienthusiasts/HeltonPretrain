trainset_path = r'/mnt/yht/data/FlickrBreeds37_Oxford_IIIT_Pet_merge/train'
validset_path = r'/mnt/yht/data/FlickrBreeds37_Oxford_IIIT_Pet_merge/valid'
nc = 37
seed = 42
log_dir = r'./log/protonet_dinov3vits_eval'
img_size = [256, 256]
load_ckpt = r'/mnt/yht/code/HeltonPretrain/log/protonet_dinov3vits_train/2025-09-25-00-34-55_train/best_val_acc.pt'

'''模型配置参数'''
model_cfgs = dict(
    type="ProtoNet",
    load_ckpt=load_ckpt,
    backbone=dict(
        type="TIMMBackbone",
        model_name="vit_small_patch16_dinov3.lvd1689m",
        pretrained=r'/mnt/yht/code/HeltonPretrain/ckpts/vit_small_patch16_dinov3.lvd1689m.pt',
        out_layers=[11],
        froze_backbone=True,
        load_ckpt=None
    ),
    head=dict(
        type="ProtoHead",
        layers_dim=[384, 256, 256], 
        nc=nc,
        cls_loss=dict(
            type="MultiClassBCELoss"
        )
    )
)
'''数据集配置参数'''
dataset_cfgs=dict(
    valid_dataset_cfg=dict(
        type="INDataset",
        img_dir=validset_path,
        mode="valid",
        img_size=img_size,
        drop_block=False
    ),
    valid_bs=1,
    num_workers=8,
    valid_shuffle=False
)

'''自定义hook'''
eval_pipeline_cfgs = dict(
    type="PretrainEvalPipeline"
)