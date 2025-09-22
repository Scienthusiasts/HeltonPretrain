# coding=utf-8
import os
import json
import torch
import logging
import datetime
import argparse
from torch import nn
from logging import Logger
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter
# 多卡并行训练:
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP





class ArgsHistory():
    """记录train或val过程中的一些变量(比如 loss, lr等) 以及tensorboard
       以字典形式存储, 例:{'loss1':[...], 'loss2':[...], ...}
    """
    def __init__(self, json_save_dir, mode):
        # NOTE:多卡
        if mode in ['train', 'eval'] or (mode=='train_ddp' and dist.get_rank() == 0):
            # tensorboard 对象
            self.tb_writer = SummaryWriter(log_dir=json_save_dir)
        self.json_save_dir = json_save_dir
        # 所有变量记录在self.args_history_dict中
        self.args_history_dict = {}


    def record(self, key, value):
        '''更新变量的记录, 变量为key, 新增的记录为value
        Args:
            - key:   要记录的当前变量的名字
            - value: 要记录的当前变量的数值
            
        Returns:
            None
        '''
        # 可能存在json格式不支持的类型, 因此统一转成float类型
        value = float(value)
        # 如果日志中还没有这个变量，则新建
        if key not in self.args_history_dict.keys():
            self.args_history_dict[key] = []
        # 更新history dict
        self.args_history_dict[key].append(value)
        # 顺便更新tensorboard
        self.tb_writer.add_scalar(key, value, len(self.args_history_dict[key]))


    def saveRecord(self):
        '''以json格式保存记录
        '''
        if not os.path.isdir(self.json_save_dir):os.makedirs(self.json_save_dir) 
        json_save_path = os.path.join(self.json_save_dir, 'args_history.json')
        # 保存
        with open(json_save_path, 'w') as json_file:
            json.dump(self.args_history_dict, json_file)


    def loadRecord(self, json_load_dir):
        '''导入上一次训练时的args(一般用于resume)
        '''
        json_path = os.path.join(json_load_dir, 'args_history.json')
        with open(json_path, "r", encoding="utf-8") as json_file:
            self.args_history_dict = json.load(json_file)




class RunnerLogger:
    """日志记录/打印相关
    """
    def __init__(self, mode:str, log_dir:str, log_interval:int, eval_interval:int, batch_num:int):
        '''生成logger日志对象用于后续打印和记录
            Args:
                mode:         网络模型
                log_dir:      日志文件保存目录
                log_interval: 日志打印间隔(几个iter打印一次)
                eval_interval:
                batch_num:    一个epoch包含的batch数量
        '''
        self.mode = mode
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.logger = logging.getLogger('runer')
        self.batch_num = batch_num
        self.logger.setLevel(level=logging.DEBUG)
        self.log_dir = log_dir
        # 日志格式
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        # 写入文件的日志
        self.log_dir = os.path.join(self.log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{mode}")
        # 日志文件保存路径
        self.log_save_path = os.path.join(self.log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{mode}.log")
        if not os.path.isdir(self.log_dir):os.makedirs(self.log_dir)
        # NOTE:多卡
        # 只在 local_rank 为 0 的进程中设置日志记录
        if mode == 'train' or (mode == 'train_ddp' and dist.get_rank() == 0):
            # 记录到log文件中的日志
            file_handler = logging.FileHandler(self.log_save_path, encoding="utf-8", mode="a")
            file_handler.setLevel(level=logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            # 终端输出的日志
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            # logger.addHandler(stream_handler)
        # 对于非主进程，可以设置一个空的日志处理器来忽略日志记录
        else:
            self.logger.addHandler(logging.NullHandler())

        self.argsHistory = ArgsHistory(self.log_dir, self.mode)


    def train_iter_log_printer(self, step:int, epoch:int, optimizer:optim, losses:dict):
        """训练/验证过程中记录/打印日志
            Args: 
                optimizer:    优化器实例(用于打印当前学习率)
                step:         当前迭代到第几个batch
                epoch:        当前迭代到第几个epoch
                losses:       当前batch的loss(字典)
            打印效果:
                2025-09-22 22:54:39,656: Epoch(train) [2][50/59]  lr: 0.000990  total_loss: 1.14722  cls_loss: 1.14722  acc: 0.82812  
                2025-09-22 22:54:42,102: Epoch(train) [2][55/59]  lr: 0.000990  total_loss: 1.08477  cls_loss: 1.08477  acc: 0.73438 
        """   
        '''记录'''
        current_lr = optimizer.param_groups[0]['lr']
        self.argsHistory.record('lr', current_lr)
        # 记录所有损失:
        for loss_name, loss_value in losses.items():
            self.argsHistory.record(loss_name, loss_value.item())

        '''打印'''
        # 每间隔log_interval个iter才打印一次
        if step % self.log_interval != 0:
            return
        # 右对齐, 打印更美观
        batch_idx = '{:>{}}'.format(step, len(f"{self.batch_num}"))
        log = ("Epoch(train) [%d][%s/%d]  lr: %8f  ") % (epoch, batch_idx, self.batch_num, current_lr)
        for loss_name, loss_value in losses.items():
            loss_log = (loss_name+": %.5f  " % (loss_value.item()))
            log += loss_log
        self.logger.info(log)



    def train_epoch_log_printer(self, epoch, evaluations, flag_metric_name):
        """训练/验证过程中记录/打印日志
            Args: 
                epoch:        当前迭代到第几个epoch
                acc:          准确率(指标)
                mAP:          mAP(指标)
                mF1_score:    mF1_score(指标)
            打印效果:
            2025-09-22 22:56:43,934: ==================================================================================================================
            2025-09-22 22:56:43,934: Epoch(valid) [1]  val_acc: 0.50286  val_mAP: 0.36976  val_mF1: 0.40191  best_val_epoch: 1  best_val_acc: 0.50286
            2025-09-22 22:56:43,934: ==================================================================================================================
        """        
        '''记录评估指标'''
        for metric_name, metric_value in evaluations.items():
            self.argsHistory.record(metric_name, metric_value)
        # 以json格式保存args
        self.argsHistory.saveRecord()

        '''打印'''
        # 找到最高准确率对应的epoch
        flag_metric_list = self.argsHistory.args_history_dict[flag_metric_name]
        best_epoch = flag_metric_list.index(max(flag_metric_list)) + 1
        best_flag_metric_val = max(flag_metric_list)
        # 打印指标
        self.logger.info('=' * 150)
        log = ("Epoch(valid) [%d]  ") % (epoch)
        for metric_name, metric_value in evaluations.items():
            loss_log = (metric_name+": %.5f  " % (metric_value))
            log += loss_log
        log += (f"best_val_epoch: {best_epoch * self.eval_interval}  best_{flag_metric_name}: {best_flag_metric_val:.5f}")
        self.logger.info(log)
        self.logger.info('=' * 150)

