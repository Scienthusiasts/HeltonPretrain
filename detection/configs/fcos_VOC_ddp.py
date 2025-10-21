train_ann_json_path = '/mnt/yht/data/VOC0712/VOC2007/Annotations/coco/train.json'
valid_ann_json_path = '/mnt/yht/data/VOC0712/VOC2007/Annotations/coco/test.json'
train_img_dir = '/mnt/yht/data/VOC0712/VOC2007/JPEGImages'
valid_img_dir = '/mnt/yht/data/VOC0712/VOC2007/JPEGImages'
# 类别名
cat_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
             "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# COCO数据集需要类别id映射
cat_maps = None

nc = 20
mode = 'train_ddp'
seed = 42
log_dir = r'./log/fcos_voc_train_ddp'
img_size = [640, 640]
epoch = 12 * 3
bs = 4
lr = 1e-4
warmup_lr = 1e-5
lr_decay = 1e-1
load_ckpt = None
log_interval = 50
eval_interval = 5
resume = None


'''模型配置参数'''
model_cfgs = dict(
    type="FCOS",
    img_size=img_size,
    nc=nc, 
    load_ckpt=load_ckpt,
    nms_score_thr=0.05,
    nms_iou_thr=0.3, 
    nms_agnostic=False,
    bbox_coder=dict(
        type="FCOSBBoxCoder",
        strides=[8, 16, 32, 64, 128]
    ),
    backbone=dict(
        type="TIMMBackbone",
        model_name="resnet50.a1_in1k",
        pretrained=False,
        out_layers=[2,3,4],
        froze_backbone=False,
        load_ckpt='ckpts/backbone_resnet50.a1_in1k.pt'
    ), 
    fpn=dict(
        type="FPN",
        in_channels=[512, 1024, 2048], 
        out_channel=256, 
        num_extra_levels=2,
    ), 
    head=dict(
        type="FCOSHead",
        nc=nc, 
        in_channel=256, 
        cnt_loss=dict(
            type="BCELoss",
            reduction="mean"
        ), 
        cls_loss=dict(
            type="FocalLoss",
            reduction="none",
            gamma=2.0, 
            alpha=0.25
        ),
        reg_loss=dict(
            type="GIoULoss",
            reduction="mean",
        ),
        assigner=dict(
            type="FCOSAssigner",
            img_size=img_size, 
            strides=[8, 16, 32, 64, 128], 
            limit_ranges=[[-1,64],[64,128],[128,256],[256,512],[512,999999]], 
            sample_radiu_ratio=1.5
        )
    )
)
'''数据集配置参数'''
dataset_cfgs=dict(
    train_dataset_cfg=dict(
        type="COCODataset",
        nc=nc, 
        cat_names=cat_names,
        ann_json_path=train_ann_json_path, 
        img_dir=train_img_dir,
        img_size=img_size, 
        mode='train', 
        mosaic_p=0.5, 
        mixup_p=0.0,
        map=cat_maps
    ),
    valid_dataset_cfg=dict(
        type="COCODataset",
        nc=nc, 
        cat_names=cat_names,
        ann_json_path=valid_ann_json_path, 
        img_dir=valid_img_dir,
        img_size=img_size, 
        mode='valid', 
        map=cat_maps
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
    type="DetectionEvalPipeline"
)