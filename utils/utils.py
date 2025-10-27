from typing import Any
import torch.nn as nn
import random
import torch
import numpy as np
import argparse
from functools import partial
import importlib
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler
import torch.distributed as dist






def multi_apply(func, *args, **kwargs) -> Any:
    """将函数应用到一组参数上(常用于多尺度预测)
      - 如果 func 每次返回多个值(tuple/list),则返回 tuple(list, list, ...)
      - 如果 func 每次返回单个值(如 Tensor),则直接返回 list

    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = list(map(pfunc, *args))
    # 没有输入时返回空列表
    if len(map_results) == 0:
        return []  

    first = map_results[0]
    # 如果每次返回的是 tuple/list -> 保持原 mmcv 行为：按位置聚合并返回 tuple(list,...)
    if isinstance(first, (tuple, list)):
        # 确保每个返回的元素长度一致会由 zip 处理
        return tuple(map(list, zip(*map_results)))
    else:
        # 单输出时，直接返回 list，便于直接赋值使用
        return list(map_results)
    



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
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    args = parser.parse_args()
    return args




def natural_key(s: str):
    # 如果类名都是数字，这会把 '10' 放到 '2' 之后；否则保持字典序
    try:
        return int(s)
    except Exception:
        return s.lower()




def to_device(batch, device, non_blocking=True):
    """递归地将所有tensor移动到指定device
    """
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    elif isinstance(batch, dict):
        return {k: to_device(v, device, non_blocking) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(to_device(v, device, non_blocking) for v in batch)
    else:
        # 非tensor类型直接返回
        return batch
    



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


def accelerate_worker_init_fn(worker_id):
    """使用accelerate封装后使用的worker_init_fn, 乌苏再手动传seed和rank
    """
    worker_info = torch.utils.data.get_worker_info()
    # 每个 worker 的基础种子来源于主进程
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)



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