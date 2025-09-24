# coding=utf-8
import os
import json
import torch
from torch import nn
from tqdm import tqdm
from PIL import Image, ImageFile
import numpy as np

from utils.metrics import *
from utils.utils import natural_key
from modules.datasets.preprocess import Transforms
# 需要import才能注册
from modules import * 
from register import MODELS



def infer_single_img(device, model, img_size, img_path, cat_names, log_dir, top_k=10):
    '''推理一张图片(经典图像分类任务)
    '''
    # 图像增强(预处理)实例
    transform = Transforms(img_size=img_size)
    model.eval()
    # 读取图像, 并进行预处理
    image = np.array(Image.open(img_path).convert('RGB'))
    tensor_img = torch.tensor(transform.valid_transform(image=image)['image'])
    tensor_img = tensor_img.permute(2,0,1).unsqueeze(0).to(device)
    # 推理
    with torch.no_grad():
        pred_logits = model(device, tensor_img)
        pred_label = torch.argmax(pred_logits, dim=1).cpu().numpy().item() 
        pred_score = pred_logits.softmax(dim=-1).cpu().numpy().squeeze()

    pred_class = cat_names[pred_label]
    pred_conf = float(pred_score[pred_label]) 

    # ========== 可视化 ==========
    # 排序类别（按置信度从大到小）
    sorted_idx = np.argsort(pred_score)[::-1]
    sorted_scores = pred_score[sorted_idx][:top_k]
    sorted_names = [cat_names[i] for i in sorted_idx[:top_k]]

    # 颜色设置：最大概率绿色，其余橙色
    colors = ['green'] + ['orange'] * (len(sorted_names)-1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # 左边：原图 + 预测文字
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Input Image")
    axes[0].text(
        5, 20, f"Pred: {pred_class} ({pred_conf:.4f})",
        fontsize=12, color="white", weight="bold",
        bbox=dict(facecolor="black", alpha=0.6, pad=3)
    )
    # 右边：横向条形图
    y_pos = np.arange(len(sorted_names))
    axes[1].barh(y_pos, sorted_scores, color=colors)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(sorted_names, fontsize=10)
    axes[1].invert_yaxis()  # 让最高概率在最上面
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel("Confidence")
    axes[1].set_title(f"Top {top_k} Class Probabilities")
    plt.tight_layout()
    plt.savefig(f"{log_dir}/infer_result.png", dpi=300)










if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = "./"
    model_cfgs = {
        "type": "FCNet",
        "load_ckpt": r'F:\Desktop\git\HeltonPretrain\log\test111\2025-09-23-01-04-07_train\best_val_acc.pt',
        "backbone":{
            "type": "TIMMBackbone",
            "model_name": "resnet50.a1_in1k",
            "pretrained": True,
            "out_layers": [4],
            "froze_backbone": True,
            "load_ckpt": None
        },
        "head":{
            "type": "MLPHead",
            "layers_dim":[2048, 256, 37], 
            "cls_loss": {
                "type": "CELoss"
            }
        }
    }
    model = MODELS.build_from_cfg(model_cfgs).to(device)
    model.eval()
    img_dir = r'F:\Desktop\master\datasets\Classification\HUAWEI_cats_dogs_fine_grained\Oxford_IIIT_Pet_FlickrBreeds\FlickrBreeds37_Oxford_IIIT_Pet_merge\valid'
    all_entries = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d)) and not d.startswith('.')]
    cat_names = sorted(all_entries, key=natural_key)
    img_size = [224, 224]
    img_path = rf"{img_dir}\pug\pug_49.jpg"
    infer_single_img(device, model, img_size, img_path, cat_names, log_dir)
