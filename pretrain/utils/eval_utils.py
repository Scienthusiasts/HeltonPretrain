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
from heltonx.utils.register import EVALPIPELINES






@EVALPIPELINES.register
class PretrainEvalPipeline():
    '''一个epoch的评估(基于验证集)
    '''
    def __call__(self, runner, model=None):
        # 直接从传入的类中获取参数(避免每个任务的特殊化):
        device = runner.device
        if model==None:
            model = runner.model
        valid_dataloader = runner.valid_dataloader
        cat_names = runner.valid_dataset.cat_names
        log_dir = runner.log_dir

        model.eval()
        # 记录真实标签和预测标签
        pred_list, true_list, soft_list = [], [], []
        # 验证时无需计算梯度
        with torch.no_grad():
            for batch_datas in tqdm(valid_dataloader):
                '''推理一个batch
                '''
                batch_datas = [v.to(device, non_blocking=True) for v in batch_datas]
                x, y = batch_datas
                pred_logits = model(x)
                # 预测结果对应置信最大的那个下标
                pred_label = torch.argmax(pred_logits, dim=1)
                # 记录(真实标签true_list, 预测标签pred_list, 置信度soft_list)
                true_list.append(y)
                pred_list.append(pred_label)
                soft_list.append(pred_logits.softmax(dim=-1))

            true_list = torch.cat(true_list, dim=0).cpu().numpy()
            pred_list = torch.cat(pred_list, dim=0).cpu().numpy()
            soft_list = torch.cat(soft_list, dim=0).cpu().numpy()
        '''评估'''
        # 准确率
        acc = sum(pred_list==true_list) / pred_list.shape[0]
        # 可视化混淆矩阵
        showComMatrix(true_list, pred_list, cat_names, log_dir)
        # 绘制PR曲线
        PRs = drawPRCurve(cat_names, true_list, soft_list, log_dir)
        # 计算每个类别的 AP, F1Score
        mAP, mF1, form = clacAP(PRs, cat_names, true_list, soft_list)
        print(form)
        print(f'acc. {acc}')
        # 评估结果以字典形式返回(统一格式, key的前缀一定有'val_')
        evaluations = dict(
            val_acc=acc, 
            val_mAP=mAP, 
            val_mF1=mF1
        )
        # 后续保存best_ckpt以val_flag_metric为参考
        flag_metric_name = "val_acc"
        return evaluations, flag_metric_name