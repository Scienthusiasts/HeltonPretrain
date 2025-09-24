import torch.nn as nn
import random
import torch
import numpy as np
import os
import argparse
import importlib
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler



def dynamic_import_class(module_path, class_name='module_name', get_class=True):
    '''动态导入类
    '''
    spec = importlib.util.spec_from_file_location(class_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if get_class:
        return getattr(module, class_name)
    else:
        return module
    


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file path')
    # 多卡
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    args = parser.parse_args()
    return args


def natural_key(s: str):
    # 如果类名都是数字，这会把 '10' 放到 '2' 之后；否则保持字典序
    try:
        return int(s)
    except Exception:
        return s.lower()

        
def init_weights(model, init_type, mean=0, std=0.01):
    '''权重初始化方法 
    '''
    for name, param in model.named_parameters():
        # 处理卷积层和全连接层的权重
        if 'weight' in name and param.dim() >= 2:
            if init_type == 'he':
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif init_type == 'normal':
                nn.init.normal_(param, mean=mean, std=std)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(param)
            elif init_type == 'uniform':
                nn.init.uniform_(param, a=-std, b=std)
        
        # 处理偏置项
        elif 'bias' in name:
            nn.init.constant_(param, 0)
        
        # 处理独立的nn.Parameter（不是模块的weight/bias）
        elif param.dim() >= 2:  # 只初始化维度>=2的Parameter
            if init_type == 'he':
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif init_type == 'normal':
                nn.init.normal_(param, mean=mean, std=std)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(param)
            elif init_type == 'uniform':
                nn.init.uniform_(param, a=-std, b=std)
        
        # 处理一维的Parameter（如偏置或标量参数）
        elif param.dim() == 1 and len(param) > 1:  # 长度>1的一维参数
            if init_type == 'normal':
                nn.init.normal_(param, mean=mean, std=std)
            elif init_type == 'uniform':
                nn.init.uniform_(param, a=-std, b=std)



def seed_everything(seed):
    '''设置全局种子
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, seed, rank=0):
    """设置Dataloader的种子
       为每个 worker 设置了一个基于初始种子和 worker ID 的独特的随机种子, 
       这样每个 worker 将产生不同的随机数序列，从而有助于数据加载过程的随机性和多样性
    """
    # rank*1000 + worker_id 避免每一个子进程数据采样重复
    worker_seed = seed + rank*1000 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)



def save_ckpt(epoch, eval_interval, model, scheduler, log_dir, argsHistory, flag_metric_name):
    '''保存权重和训练断点
        Args:
            - epoch:       当前epoch
            - model:       网络模型实例
            - scheduler:   学习率策略实例(包含优化器)
            - log_dir:     日志文件保存目录
            - argsHistory: 日志文件记录实例
            - logger:      日志输出实例

        Returns:
            None
    '''  
    # ckpt一定不包含ddp那层封装的module
    ckpt = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    # checkpoint_dict能够恢复断点训练
    checkpoint_dict = {
        'epoch': epoch, 
        'model_state_dict': model.state_dict(), 
        'optim_state_dict': scheduler.optimizer.state_dict(),
        'sched_state_dict': scheduler.base_scheduler.state_dict()
        }
    torch.save(checkpoint_dict, os.path.join(log_dir, f"train_epoch{epoch}.pt"))
    torch.save(ckpt, os.path.join(log_dir, "last.pt"))
    # 如果本次Epoch的参考指标最大，则保存网络参数
    flag_metric_list = argsHistory.args_history_dict[flag_metric_name]
    best_flag_metric_val = max(flag_metric_list)
    best_epoch = flag_metric_list.index(best_flag_metric_val) + 1
    if epoch == best_epoch * eval_interval:
        torch.save(ckpt, os.path.join(log_dir, f'best_{flag_metric_name}.pt'))




def train_resume(resume, model, optimizer, scheduler, runner_logger, batch_nums):
    '''保存权重和训练断点
        Args:
            resume:      是否恢复断点训练
            model:       网络模型
            optimizer:   优化器
            runner_logger:      日志输出实例
            scheduler:
            batch_nums:
        Returns:
            None
    '''  
    ckpt = torch.load(resume, map_location="cpu")
    resume_epoch = ckpt['epoch'] + 1
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optim_state_dict'])
    scheduler.base_scheduler.load_state_dict(ckpt['sched_state_dict'])
    runner_logger.logger.info(f'resume:{resume}')
    runner_logger.logger.info(f'resume_epoch:{resume_epoch}')
    # 导入上一次中断训练时的args
    json_dir, _ = os.path.split(resume)
    runner_logger.argsHistory.loadRecord(json_dir)
    scheduler.last_epoch = batch_nums * resume_epoch
    return resume_epoch




def set_dataloader_epoch(dataloader, epoch, base_seed):
    """保证 DataLoader resume 后的随机性与原训练一致
    Args:
        dataloader: DataLoader 对象
        epoch:      当前 epoch
        base_seed:  训练时固定的基础随机种子
    """
    # 处理 DistributedSampler
    # DDP时, 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果
    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    # 处理普通 DataLoader shuffle
    elif hasattr(dataloader, "sampler"):
        # 如果使用了 RandomSampler，说明启用了 shuffle
        from torch.utils.data import RandomSampler
        if isinstance(dataloader.sampler, RandomSampler):
            seed = base_seed + epoch
            seed_everything(seed)