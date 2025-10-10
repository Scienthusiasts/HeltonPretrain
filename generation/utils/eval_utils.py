# coding=utf-8
import os
import json
import torch
from torch import nn
from tqdm import tqdm
from PIL import Image, ImageFile
import numpy as np

from pretrain.utils.metrics import *
from pretrain.datasets.preprocess import Transforms
# 多卡并行训练:
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.register import EVALPIPELINES


@EVALPIPELINES.register
class GenerationEvalPipeline():
    '''一个epoch的评估(基于验证集)
    '''
    def __call__(self, runner):
        # 直接从传入的类中获取参数(避免每个任务的特殊化):
        model = runner.model
        epoch = runner.epoch
        log_dir = runner.log_dir

        # 图像均值 标准差
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([[0.229, 0.224, 0.225]]) 

        model.eval()
        # 图像生成
        samples = model(bs=25, return_loss=False)

        # 可视化
        generate_images = samples[-1]
        B, C, H, W = generate_images.shape
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))  # Create an 8x8 grid of subplots
        for i, ax in enumerate(axes.flat):
            gen_img_norm = generate_images[i].reshape(C, H, W).transpose((1,2,0))
            # figtest = reverse_transform(torch.from_numpy(generate_image))
            gen_img = gen_img_norm * std + mean
            ax.imshow(gen_img) 
            ax.axis("off")  

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"epoch_{epoch}.png"), dpi=300)

        # TODO:
        # 评估结果以字典形式返回(统一格式, key的前缀一定有'val_')
        evaluations = dict(
            ssim=epoch
        )
        # 后续保存best_ckpt以val_flag_metric为参考
        flag_metric_name = "ssim"
        return evaluations, flag_metric_name