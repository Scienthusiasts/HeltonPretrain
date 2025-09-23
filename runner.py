# coding=utf-8
import os
import json
import torch
import shutil
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from functools import partial
from torch.utils.data import DataLoader

from utils.utils import seed_everything, worker_init_fn, save_ckpt
from utils.log_utils import *
from utils.eval_utils import eval_epoch
from utils.hooks import hook_after_batch, hook_after_epoch
# 需要import才能注册
from modules import * 
from register import MODELS, DATASETS, OPTIMIZERS




class Runner():
    """整合训练/验证/推理时的抽象流程"""
    def __init__(self, mode, epoch, seed, log_dir, log_interval, eval_interval, model_cfgs, dataset_cfgs, optimizer_cfgs, scheduler_cfgs):
        """初始化各种模块
            Args:
                mode:           train, train_ddp, eval, test, ...
                epoch:          训练多少轮
                seed:           全局种子
                log_dir:
                log_interval:
                eval_interval:
                model_cfgs:     和网络模型有关的配置参数
                dataset_cfgs:   和数据集有关的配置参数
                optimizer_cfgs: 和优化器有关的配置参数
                scheduler_cfgs: 和学习率衰减策略有关的配置参数

        """
        self.mode = mode
        self.log_dir = log_dir
        self.eval_interval = eval_interval
        # GPU/CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epoch = epoch
        self.cur_epoch = 0
        self.cur_step = 0
        self.losses = None
        self.seed = seed
        # 设置全局种子
        seed_everything(self.seed)
                                           
        '''导入网络'''
        self.model = MODELS.build_from_cfg(model_cfgs).to(self.device)
        print(self.model)

        '''导入数据集'''
        self.train_dataset = DATASETS.build_from_cfg(dataset_cfgs["train_dataset_cfg"])
        self.valid_dataset = DATASETS.build_from_cfg(dataset_cfgs["valid_dataset_cfg"])
        print(f'trainset图像数:{self.train_dataset.__len__()} validset图像数:{self.valid_dataset.__len__()}')
        print(f'trainset类别数:{self.train_dataset.get_cls_num()} validset类别数:{self.valid_dataset.get_cls_num()}')
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=dataset_cfgs["train_bs"],
            num_workers=dataset_cfgs["num_workers"],
            shuffle=dataset_cfgs["train_shuffle"],
            collate_fn=self.train_dataset.dataset_collate,
            worker_init_fn=partial(worker_init_fn, seed=self.seed)
        )
        self.valid_dataloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=dataset_cfgs["valid_bs"],
            num_workers=dataset_cfgs["num_workers"],
            shuffle=dataset_cfgs["valid_shuffle"],
            collate_fn=self.valid_dataset.dataset_collate,
            worker_init_fn=partial(worker_init_fn, seed=self.seed)
        )
        # 一个epoch包含多少batch
        self.train_batch_num = len(self.train_dataloader)

        '''优化器'''
        self.optimizer = OPTIMIZERS.build_from_cfg(optimizer_cfgs, params=self.model.parameters())
        # 学习率衰减策略(+warmup)
        scheduler_cfgs["base_schedulers_cfgs"]["step_size"] *= self.train_batch_num
        scheduler_cfgs["warmup_schedulers_cfgs"]["warmup_epochs"] *= self.train_batch_num
        base_scheduler = SCHEDULERS.build_from_cfg(scheduler_cfgs["base_schedulers_cfgs"], optimizer=self.optimizer)
        self.scheduler = SCHEDULERS.build_from_cfg(scheduler_cfgs["warmup_schedulers_cfgs"], base_scheduler=base_scheduler, optimizer=self.optimizer)

        '''日志模块'''
        if mode in ['train', 'train_ddp', 'eval']:
            self.runner_logger = RunnerLogger(self.mode, self.log_dir, log_interval, eval_interval, self.train_batch_num)
            self.log_dir = self.runner_logger.log_dir

        '''Hook 管理'''
        self._hooks = {}

    # ========== Hook 机制 ==========
    def register_hook(self, event: str, func):
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(func)

    def call_hooks(self, event: str, *args, **kwargs):
        for hook in self._hooks.get(event, []):
            hook(*args, **kwargs)


    def fit_batch(self, batch_datas):
        """一个batch的训练流程(前向+反向)
            Args:
                batch_datas: dataloader传来的数据+标签
        """
        self.call_hooks("before_batch", runner=self)

        # 一个batch的前向传播+计算损失 
        self.losses = self.model(self.device, batch_datas, return_loss=True)
        # 将上一次迭代计算的梯度清零 
        self.optimizer.zero_grad()
        # 反向传播
        self.losses['total_loss'].backward()
        # 更新权重
        self.optimizer.step()

        self.call_hooks("after_batch", runner=self)



    def fit_epoch(self):
        '''一个epoch的训练
        '''
        self.call_hooks("before_epoch", runner=self)

        self.model.train()
        for step, batch_datas in enumerate(self.train_dataloader):
            self.cur_step = step
            '''一个batch的训练'''
            self.fit_batch(batch_datas)
            # 一个batch结束后更新学习率
            self.scheduler.step() 

        self.call_hooks("after_epoch", runner=self)



    def fit(self):
        '''所有epoch的训练流程(训练+验证)
        '''
        self.call_hooks("before_fit", runner=self)

        for epoch in range(1, self.epoch+1):
            self.cur_epoch = epoch
            '''一个epoch的训练'''
            self.fit_epoch()

        self.call_hooks("after_fit", runner=self)









# for test only:
if __name__ == '__main__':
    mode = 'train'
    epoch = 50
    seed = 42
    log_dir = "./log/test111"
    log_interval = 10
    eval_interval = 1

    # 模型配置参数
    # model_cfgs = {
    #     "type": "MSF2Net",
    #     "load_ckpt": None,
    #     "backbone":{
    #         "type": "TIMMBackbone",
    #         "model_name": "resnet50.a1_in1k",
    #         "pretrained": True,
    #         "out_layers": [1, 2, 3, 4],
    #         "froze_backbone": True,
    #         "load_ckpt": None
    #     },
    #     "head":{
    #         "type": "MLPHead",
    #         "layers_dim":[256+512+1024+2048, 256, 37], 
    #         "cls_loss": {
    #             "type": "CELoss"
    #         }
    #     }
    # }
    model_cfgs = {
        "type": "FCNet",
        "load_ckpt": None,
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
    # 数据集配置参数
    dataset_cfgs = {
        "train_dataset_cfg": {
            "type": "INDataset",
            "img_dir": r'F:\Desktop\master\datasets\Classification\HUAWEI_cats_dogs_fine_grained\The_Oxford_IIIT_Pet_Dataset\images\train',
            "mode": "train",
            "img_size": [224, 224],
            "drop_block": True
        },
        "valid_dataset_cfg": {
            "type": "INDataset",
            "img_dir": r'F:\Desktop\master\datasets\Classification\HUAWEI_cats_dogs_fine_grained\The_Oxford_IIIT_Pet_Dataset\images\valid',
            "mode": "train",
            "img_size": [224, 224],
            "drop_block": False
        },
        "train_bs": 64,
        "valid_bs": 1,
        "num_workers": 0,
        "train_shuffle": True,
        "valid_shuffle": False
    }

    # 优化器配置参数
    optimizer_cfgs = {
        "type": "AdamW",
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "weight_decay": 0.01
    }
    # 学习率衰减策略配置参数
    scheduler_cfgs = {
        "base_schedulers_cfgs": {
            "type": "StepLR",
            # 每间隔step_size个epoch更新学习率
            "step_size": 1,
            # 每次学习率变为原来的gamma倍
            "gamma": 0.1**(1/epoch),
        },
        "warmup_schedulers_cfgs": {
                "type": "WarmupScheduler",
                "min_lr": 1e-5,
                "warmup_epochs": 1
        }
    }
    runner = Runner(mode, epoch, seed, log_dir, log_interval, eval_interval, model_cfgs, dataset_cfgs, optimizer_cfgs, scheduler_cfgs)
    # 注册 Hook
    runner.register_hook("after_batch", hook_after_batch)
    runner.register_hook("after_epoch", hook_after_epoch)
    # 训练
    runner.fit()





