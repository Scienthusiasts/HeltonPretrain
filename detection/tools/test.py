# coding=utf-8
import os
import json
import torch
from torch import nn
from tqdm import tqdm
from PIL import Image, ImageFile
import numpy as np
from collections import Counter
from detection.utils.metrics import *
from detection.utils.utils import OpenCVDrawBox
from utils.register import EVALPIPELINES
from utils.utils import to_device
from detection.datasets.preprocess import Transforms
from utils.register import MODELS










def infer_single_img(model, device, img_path, cat_names, save_vis_path):
    '''
    推理一张图
        Args:
            device:        cpu/cuda
            img_size:      固定图像大小 如[832, 832]
            img_path:      图像路径
            save_vis_path: 可视化图像保存路径

        Returns:
            boxes:       网络回归的box坐标    [obj_nums, 4]
            box_scores:  网络预测的box置信度  [obj_nums]
            box_classes: 网络预测的box类别    [obj_nums]
    '''
    # 图像均值 标准差
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([[0.229, 0.224, 0.225]]) 
    transform = Transforms(img_size=img_size)

    image = np.array(Image.open(img_path).convert('RGB'))
    tensor_img = torch.tensor(transform.test_transform(image=image)['image'])
    resize_img = ((tensor_img.numpy() * std + mean) * 255).astype(np.uint8)
    tensor_img = tensor_img.permute(2,0,1).unsqueeze(0).to(device)

    '''每个类别都获得一个随机颜色'''
    image2color = dict()
    for cat in cat_names:
        image2color[cat] = (np.random.random((1, 3)) * 0.7 + 0.3).tolist()[0]

    '''推理一张图像'''
    boxes, box_scores, box_classes = model.infer(tensor_img)
    #  检测出物体才继续    
    if len(boxes) == 0: 
        print(f'no objects in image: {img_path}.')
        return boxes, box_scores, box_classes

    '''画框'''
    resize_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2BGR)
    resize_img = OpenCVDrawBox(resize_img, boxes, box_classes, box_scores, save_vis_path, image2color, cat_names, resize_size=[2000, 2000], show_text=True)
    cv2.imwrite(save_vis_path, resize_img)
    # 统计检测出的类别和数量
    detect_cls = dict(Counter(box_classes))
    detect_name = {}
    for key, val in detect_cls.items():
        detect_name[cat_names[key]] = val
    print(f'detect result: {detect_name}')






if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # cat_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
    #             "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    cat_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    nc = len(cat_names)

    img_size = [800, 800]
    load_ckpt = 'log/fcos_coco_train_ddp/2025-10-21-16-36-31_train_ddp/last.pt'

    '''模型配置参数'''
    model_cfgs = dict(
        type="FCOS",
        img_size=img_size,
        nc=nc, 
        load_ckpt=load_ckpt,
        nms_score_thr=0.2,
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
                sample_radiu_ratio=1.5,
            )
        )
    )
    model = MODELS.build_from_cfg(model_cfgs).to(device)
    model.eval()
    img_path = 'detection/demos/13.jpg'
    save_vis_path = './det_res.jpg'
    infer_single_img(model, device, img_path, cat_names, save_vis_path)