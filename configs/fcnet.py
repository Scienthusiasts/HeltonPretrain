trainset_path = r'/mnt/yht/data/FlickrBreeds37_Oxford_IIIT_Pet_merge/train'
validset_path = r'/mnt/yht/data/FlickrBreeds37_Oxford_IIIT_Pet_merge/valid'
nc = 37
mode = 'train_ddp'
seed = 42
log_dir = r'./log/fcnet_train_ddp'
img_size = [224, 224]
epoch = 50
bs=16 # 64
lr = 1e-3
warmup_lr = 1e-5
lr_decay = 1e-1
# load_ckpt = r'/mnt/yht/code/HeltonPretrain/log/test111/2025-09-23-12-35-44_train/best_val_acc.pt'
load_ckpt = None
log_interval = 50
eval_interval = 1


'''模型配置参数'''
model_cfgs = dict(
    type="FCNet",
    load_ckpt=load_ckpt,
    backbone=dict(
        type="TIMMBackbone",
        model_name="resnet50.a1_in1k",
        pretrained=True,
        out_layers=[4],
        froze_backbone=False,
        load_ckpt=None
    ),
    head=dict(
        type="MLPHead",
        layers_dim=[2048, 256, nc], 
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