json_data_path = '/data/yht/data/llm/pretrain_hq.jsonl'
huggingface_weights_dir = 'ckpts/hugging_face/MiniMind2-R1'

mode = 'train_ddp'
seed = 42
log_dir = r'./log/llm/minimind_pretrain'
epoch = 12
bs = 32
lr = 5e-4
warmup_lr = 1e-5
lr_decay = 1e-1
load_ckpt = None
log_interval = 50
eval_interval = 1
resume = None
# 梯度累加策略, bs等效于 bs*grad_accumulate
grad_accumulate=4
# 梯度裁剪策略
grad_clip=1



'''模型配置参数'''
model_cfgs = dict(
    type="PretrainLLM",
    load_ckpt=load_ckpt, 
    hidden_size=768, 
    num_hidden_layers=16, 
    use_moe=False, 
    inference_rope_scaling=False,
    loss=dict(
        type="CELoss",
        reduction='none'
    )
)
'''数据集配置参数'''
dataset_cfgs=dict(
    train_dataset_cfg=dict(
        type="PretrainDataset",
        json_data_path=json_data_path, 
        huggingface_weights_dir=huggingface_weights_dir, 
        max_length=512
    ),
    valid_dataset_cfg=None,
    train_bs=bs,
    num_workers=8,
    train_shuffle=True,
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

'''任务特定的评估pipeline'''
# eval_pipeline_cfgs = None