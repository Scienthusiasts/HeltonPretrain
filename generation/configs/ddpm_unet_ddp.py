# trainset_path = r'/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images'
trainset_path = r'/mnt/yht/data/FlickrBreeds37_Oxford_IIIT_Pet_merge'
mode = 'train_ddp'
seed = 42
log_dir = r'./log/ddpn_unet_train_ddp'
img_size = [128, 128]
epoch = 200
bs = 24
lr = 2e-4
warmup_lr = lr*1e-2
lr_decay = 1e-1
load_ckpt = None
log_interval = 10
eval_interval = 1
resume = None


'''模型配置参数'''
model_cfgs = dict(
    type="DDPM",
    img_size=img_size,
    batch_size=bs,
    load_ckpt=load_ckpt,
    schedule_name="linear_beta_schedule",
    timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    loss_type='huber',
    denoise_model=dict(
        type="UNet",
        dim=img_size[0],
        channels=3,
        dim_mults=(1, 2, 4,)
    )
)
'''数据集配置参数'''
dataset_cfgs=dict(
    train_dataset_cfg=dict(
        type="GenDataset",
        img_dir=trainset_path,
        img_size=img_size,
    ),
    valid_dataset_cfg=None,
    train_bs=bs,
    num_workers=8,
    train_shuffle=True
)
'''优化器配置参数'''
optimizer_cfgs=dict(
    type="Adam",
    lr=lr
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

'''任务特定的评估pipeline'''
eval_pipeline_cfgs = dict(
    type="GenerationEvalPipeline"
)