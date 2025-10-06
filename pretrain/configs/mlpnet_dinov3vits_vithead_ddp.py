# trainset_path = r'/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/train'
# validset_path = r'/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/valid'
trainset_path = r'/mnt/yht/data/FlickrBreeds37_Oxford_IIIT_Pet_merge/train'
validset_path = r'/mnt/yht/data/FlickrBreeds37_Oxford_IIIT_Pet_merge/valid'
nc = 37
mode = 'train_ddp'
seed = 42
log_dir = r'./log/mlpnet_dinov3vits_vithead_train_ddp'
img_size = [224, 224]
epoch = 50
bs = 16
lr = 1e-3  # 使用transformer结构的head好像学习率不能太大, 否则容易训崩
warmup_lr = 1e-5
lr_decay = 1e-1
load_ckpt = None
log_interval = 50
eval_interval = 1
resume = None


'''模型配置参数'''
model_cfgs = dict(
    type="MLPNet",
    load_ckpt=load_ckpt,
    backbone=dict(
        type="TIMMBackbone",
        model_name="vit_small_patch16_dinov3.lvd1689m",
        pretrained=False,
        out_layers=[11],
        froze_backbone=True,
        load_ckpt=r'/mnt/yht/code/HeltonPretrain/ckpts/backbone_vit_small_patch16_dinov3.lvd1689m.pt'
    ),
    head=dict(
        type="ViTHead",
        nc=nc, 
        in_dim=384, 
        num_heads=8, 
        mlp_ratio=4.0, 
        dropout=0.0,
        cls_loss=dict(
            type="CELoss"
        )
    )
)
'''数据集配置参数'''
dataset_cfgs=dict(
    train_dataset_cfg=dict(
        type="INDataset",
        img_dir=trainset_path,
        mode="train",
        img_size=img_size,
        drop_block=True
    ),
    valid_dataset_cfg=dict(
        type="INDataset",
        img_dir=validset_path,
        mode="valid",
        img_size=img_size,
        drop_block=False
    ),
    train_bs=bs,
    valid_bs=1,
    num_workers=8,
    train_shuffle=True,
    valid_shuffle=False
)
'''优化器配置参数'''
optimizer_cfgs=dict(
    type="AdamW",
    lr=lr,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
'''学习率衰减策略配置参数'''
scheduler_cfgs=dict(
    base_schedulers_cfgs=dict(
        type="StepLR",
        # 每间隔step_size个epoch更新学习率
        step_size=1,
        # 每次学习率变为原来的gamma倍
        gamma=lr_decay**(1/epoch),
    ),
    warmup_schedulers_cfgs=dict(
            type="WarmupScheduler",
            min_lr=warmup_lr,
            warmup_epochs=1
    )
)