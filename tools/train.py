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

from utils.utils import seed_everything, worker_init_fn, get_args, dynamic_import_class, set_dataloader_epoch
from utils.ckpts_utils import train_resume
from utils.log_utils import *
from utils.hooks import NecessaryHook
# 需要import才能注册
from pretrain import * 
from generation import * 
from optimization import *
from utils.register import MODELS, DATASETS, OPTIMIZERS, SCHEDULERS, EVALPIPELINES



class Trainer():
    """整合训练/验证/推理时的抽象流程"""
    def __init__(self, mode, epoch, seed, log_dir, log_interval, eval_interval, resume_path, model_cfgs, dataset_cfgs, optimizer_cfgs, scheduler_cfgs):
        """初始化各种模块
            Args:
                mode:               train, train_ddp
                epoch:              训练多少轮
                seed:               全局种子
                log_dir:
                log_interval:
                eval_interval:
                resume_path:
                model_cfgs:         和网络模型有关的配置参数
                dataset_cfgs:       和数据集有关的配置参数
                optimizer_cfgs:     和优化器有关的配置参数
                scheduler_cfgs:     和学习率衰减策略有关的配置参数
        """
        self.mode = mode
        self.log_dir = log_dir
        self.eval_interval = eval_interval
        self.epoch = epoch
        self.cur_epoch = 1
        self.start_epoch = 1
        self.cur_step = 0
        self.losses = None
        self.seed = seed
        # 设置全局种子
        seed_everything(self.seed)

        '''确定 CPU/单卡/DDP 训练策略'''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.mode=='train_ddp':
            self.local_rank = int(os.environ["LOCAL_RANK"]) 
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group('nccl')

        '''导入网络'''
        self.model = MODELS.build_from_cfg(model_cfgs)
        # self.model = torch.compile(self.model)
        if self.mode=='train_ddp':
            # 多卡时同步BN
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).cuda(self.local_rank)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], find_unused_parameters=False)
        else:
            self.model.to(self.device)

        '''导入数据集'''
        self.train_dataset = DATASETS.build_from_cfg(dataset_cfgs["train_dataset_cfg"])
        self.valid_dataset = DATASETS.build_from_cfg(dataset_cfgs["valid_dataset_cfg"]) if dataset_cfgs["valid_dataset_cfg"] else None
        # DDP训练时需要sampler且shuffle=False
        train_sampler = None
        if self.mode == 'train_ddp':
            train_sampler = DistributedSampler(self.train_dataset)
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            sampler=train_sampler,
            batch_size=dataset_cfgs["train_bs"],
            num_workers=dataset_cfgs["num_workers"],
            shuffle=False if self.mode == 'train_ddp' else dataset_cfgs["train_shuffle"],
            collate_fn=self.train_dataset.dataset_collate,
            worker_init_fn=partial(worker_init_fn, seed=self.seed),
            pin_memory=True # CPU → GPU 数据拷贝速度加速
        )
        self.valid_dataloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=dataset_cfgs["valid_bs"],
            num_workers=dataset_cfgs["num_workers"],
            shuffle=dataset_cfgs["valid_shuffle"],
            collate_fn=self.valid_dataset.dataset_collate,
            worker_init_fn=partial(worker_init_fn, seed=self.seed),
            pin_memory=True # CPU → GPU 数据拷贝速度加速
        ) if dataset_cfgs["valid_dataset_cfg"] else None
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
        self.runner_logger = RunnerLogger(self.mode, self.log_dir, log_interval, eval_interval, self.train_batch_num)
        self.log_dir = self.runner_logger.log_dir

        '''Hook 管理'''
        self._hooks = {}

        # resume
        if resume_path:
            self.start_epoch = train_resume(resume_path, self.model, self.optimizer, self.scheduler, self.runner_logger, self.train_batch_num)
         # 打印模型详细信息
        if self.mode == 'train' or (self.mode == 'train_ddp' and dist.get_rank() == 0):
            self.runner_logger.log_model_info(self.model, self.optimizer)

    # Hook 机制 ==========
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



    def fit_batch(self, batch_datas):
        """一个batch的训练流程(前向+反向)
            Args:
                batch_datas: dataloader传来的数据+标签
        """
        self.call_hooks("before_batch", runner=self)

        # 一个batch的前向传播+计算损失 
        # 确保 batch_datas 的所有数据已经在 self.device 上(batch_datas的组织形式是list)
        batch_datas = [v.to(self.device, non_blocking=True) for v in batch_datas]
        self.losses = self.model(batch_datas, return_loss=True)
        # 将上一次迭代计算的梯度清零 
        self.optimizer.zero_grad()
        # 反向传播
        self.losses["total_loss"] = sum(
            v for v in self.losses.values()
            if torch.is_tensor(v) and v.requires_grad
        )
        self.losses['total_loss'].backward()
        # 更新权重
        self.optimizer.step()

        self.call_hooks("after_batch", runner=self)



    def fit_epoch(self):
        '''一个epoch的训练
        '''
        self.call_hooks("before_epoch", runner=self)

        self.model.train()
        # 固定每个epoch的随机性:
        set_dataloader_epoch(self.train_dataloader, self.cur_epoch, self.seed)
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

        for epoch in range(self.start_epoch, self.epoch+1):
            self.cur_epoch = epoch
            '''一个epoch的训练'''
            self.fit_epoch()

        self.call_hooks("after_fit", runner=self)



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
    runner = Trainer(cargs.mode, cargs.epoch, cargs.seed, cargs.log_dir, cargs.log_interval, cargs.eval_interval, cargs.resume, 
                     cargs.model_cfgs, cargs.dataset_cfgs, cargs.optimizer_cfgs, cargs.scheduler_cfgs)
    # 注册 Hook
    # 任务特定的评估pipeline
    eval_pipeline = EVALPIPELINES.build_from_cfg(cargs.eval_pipeline_cfgs)
    hook = NecessaryHook(eval_pipeline)
    runner.register_hook("after_batch", hook.hook_after_batch)
    runner.register_hook("after_epoch", hook.hook_after_epoch)
    runner.register_hook("after_eval",  hook.hook_after_eval)

    if cargs.mode in ['train', 'train_ddp']:
        # 拷贝一份当前训练对应的config文件(方便之后查看细节)
        shutil.copy(config_path, os.path.join(runner.log_dir, os.path.basename(config_path)))
        runner.fit()





