# trainset_path = r'/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/train'
# validset_path = r'/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/valid'
trainset_path = r'/mnt/yht/data/FlickrBreeds37_Oxford_IIIT_Pet_merge/train'
validset_path = r'/mnt/yht/data/FlickrBreeds37_Oxford_IIIT_Pet_merge/valid'
nc = 37
mode = 'train_ddp'
seed = 42
log_dir = r'./log/mlpnet_multi_tasks_distill_train'
img_size = [224, 224]
clip_emb_dim = 512
clip_pretrain_path = r'/mnt/yht/code/HeltonPretrain/ckpts/hugging_face/models--qihoo360--fg-clip-base/snapshots/f30d2b82ba939fd54ca732426f99f4d6c3c92387'
# img_size = [336, 336]
# clip_emb_dim = 768
# clip_pretrain_path = r'/mnt/yht/code/HeltonPretrain/ckpts/hugging_face/models--qihoo360--fg-clip-large/snapshots/19c2df7667052518ade09341652562b89b1332da'

epoch = 50
bs = 32
lr = 1e-3
warmup_lr = 1e-5
lr_decay = 1e-1
load_ckpt = None
log_interval = 50
eval_interval = 1
resume = None


template_prompt = [
        "a {} in the scene."
        "A {} with a happy expression.",
        "A photo of a friendly {}.",
        "A lovely little {}.",
        "a photo showing a {} in the scene",
        "a color picture of a {}, it is cute",
        "a photograph of a nice {}.",
        "a cropped photo of a {}, it is playful.",
        "I own a {} and I really like it.",
        "a picture of a {} taken long time ago.",
        "the picture showing a {} in the center.",
        "a picture of one {} in the scene.", # 
        "I adopted this {} several years ago.",
        "I took a picture of my {}.",
        "I love my {} and it loves me too.",
        "The {} in the picture is my friend's.",
        "This {} was a birthday present from my best friend.",
        "I accidentally snapped a picture of this {}.",
        "I petted my {} and she seemed to enjoy it.",
        "I called out to the {} and it rushed to me.",
        "My {} looking at the camera. It's the best memory ever.",
        "this {} used to be my best mate. Now it's gone.",
        "You're the best, my good {}.",
        "the {} is staring at me, want something to eat.",
        "My neighbour's {}, it looks funny.",
        "An elegant {} posing gracefully for the camera.",
        "This {} always greets me when I come home from work.",
        "The {} I rescued from the shelter last month.",
        "My {} waiting patiently for dinner time.",
        "The {} that helped me through difficult times.",
        "A {} communi{}ing through its expressive body language.",
        "this {}, A loyal companion with eyes full of devotion.",
        "A mischievous little {}, always getting into something new.",
        "Caught in the act! My {} staring guiltily at the mess it just made.",
        "my {}, The best welcome home committee, always waiting at the door.", 
        "My shadow, my {}, following me from room to room throughout the day.",
        "A blur of energy, the {} racing across the yard at full speed.",
        "A candid moment, this {} completely unaware of the camera.",
        "A calming influence, this {} knowing how to soothe a bad day.",
        "An unexpected friendship that grew into an unbreakable bond, my {}.",
        "The simple pleasure of watching my {} enjoy the sunshine.",
        "The first picture I took before we brought the {} home." 
    ]
cat_names = ['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'Bengal', 'Birman', 
             'Bombay', 'boxer', 'British_Shorthair', 'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter', 
             'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'Maine_Coon', 
             'miniature_pinscher', 'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed', 
             'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']



'''模型配置参数'''
model_cfgs = dict(
    type="MLPNetMultiTasks",
    load_ckpt=load_ckpt,
    ensemble_pred=True,
    backbone=dict(
        type="TIMMBackbone",
        model_name="resnet50.a1_in1k",
        pretrained=False,
        out_layers=[4],
        froze_backbone=False,
        load_ckpt=r'/mnt/yht/code/HeltonPretrain/ckpts/backbone_resnet50.a1_in1k.pt'
    ),
    cls_head=dict(
        type="MLPHead",
        layers_dim=[2048, 1024, nc], 
        cls_loss=dict(
            type="CELoss",
            reduction='mean'
        )
    ),
    emb_head=dict(
        type="CLIPDistillEmbHead",
        layers_dim=[2048, 1024, clip_emb_dim], 
        clip_model=dict(
            type="Qihoo360FGCLIP",
            pretrain_path=clip_pretrain_path,
        ),
        distill_loss=dict(
            type="SmoothL1Loss",
            reduction='mean'
        ),
        contrast_loss=dict(
            type="CELoss",
            reduction='mean'
        ),
        cat_names=cat_names,
        template_prompt=template_prompt,
    ),
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

'''任务特定的评估pipeline'''
eval_pipeline_cfgs = dict(
    type="PretrainEvalPipeline"
)