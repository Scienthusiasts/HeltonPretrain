# coding=utf-8
import os
import json
import torch
import shutil
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
from utils.utils import seed_everything, worker_init_fn, get_args, dynamic_import_class
from utils.log_utils import *
from pretrain.utils.hooks import hook_after_eval
# 需要import才能注册
from pretrain import * 
from utils.register import MODELS, DATASETS




class Evaler():
    """整合训练/验证/推理时的抽象流程"""
    def __init__(self, seed, log_dir, model_cfgs, dataset_cfgs):
        """初始化各种模块
            Args:
                seed:           全局种子
                log_dir:
                model_cfgs:     和网络模型有关的配置参数
                dataset_cfgs:   和数据集有关的配置参数

        """
        self.log_dir = log_dir
        self.seed = seed
        self.cur_epoch = 1
        # 设置全局种子
        seed_everything(self.seed)
        # GPU/CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        '''导入网络'''
        self.model = MODELS.build_from_cfg(model_cfgs).to(self.device)

        '''导入数据集'''
        self.valid_dataset = DATASETS.build_from_cfg(dataset_cfgs["valid_dataset_cfg"])
        print(f'validset图像数:{self.valid_dataset.__len__()} validset类别数:{self.valid_dataset.get_cls_num()}')
        self.valid_dataloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=dataset_cfgs["valid_bs"],
            num_workers=dataset_cfgs["num_workers"],
            shuffle=dataset_cfgs["valid_shuffle"],
            collate_fn=self.valid_dataset.dataset_collate,
            worker_init_fn=partial(worker_init_fn, seed=self.seed),
            pin_memory=True # CPU → GPU 数据拷贝速度加速
        )

        '''日志模块'''
        self.runner_logger = RunnerLogger('eval', self.log_dir, 1, 1, 1)
        self.log_dir = self.runner_logger.log_dir

        '''Hook 管理'''
        self._hooks = {}


    def register_hook(self, event: str, func):
        """注册hook
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(func)


    def call_hooks(self, event: str, *args, **kwargs):
        """调用一个hook
        """
        for hook in self._hooks.get(event, []):
            hook(*args, **kwargs)


    def eval(self):
        '''验证流程
        '''
        self.call_hooks("before_eval", runner=self)
        self.call_hooks("after_eval", runner=self)




# for test only:
if __name__ == '__main__':
    args = get_args()
    config_path = args.config
    # 使用动态导入模块导入参数文件
    cargs = dynamic_import_class(config_path, get_class=False)
    # 初始化runner
    runner = Evaler(cargs.seed, cargs.log_dir, cargs.model_cfgs, cargs.dataset_cfgs)
    # 注册 Hook
    runner.register_hook("after_eval", hook_after_eval)
    runner.eval()





