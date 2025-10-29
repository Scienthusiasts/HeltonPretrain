import torch.nn as nn
import random
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler
import torch.distributed as dist
import pickle







class NoSaveWrapper(nn.Module):
    """使用了这个包装器的nn.Module模块, 在保存权重时不会保持该模块的权重
       (常用于蒸馏模型的teacher)
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        # 不保存
        return {}  

    def _load_from_state_dict(self, *args, **kwargs):
        # 不加载
        return  







class DDPSafeDataset():
    
    def ddp_safe_load(self, load_fn, verbose=True):
        if not dist.is_initialized():
            if verbose:
                print("[Single] Loading dataset directly...")
            return load_fn()

        rank = dist.get_rank()
         # 🚀 关键修改：根据 rank 放到对应 GPU
        device = torch.device("cuda", rank) 

        if rank == 0:
            if verbose:
                print(f"[Rank 0] Loading dataset ...")
            data = load_fn()
            buffer = pickle.dumps(data)
            buffer_tensor = torch.ByteTensor(list(buffer)).to(device)
            length_tensor = torch.tensor([len(buffer_tensor)], dtype=torch.long, device=device)
        else:
            data = None
            buffer_tensor = torch.ByteTensor().to(device)
            length_tensor = torch.tensor([0], dtype=torch.long, device=device)

        # 广播长度
        dist.broadcast(length_tensor, src=0)
        # 分配同样大小的tensor
        if rank != 0:
            buffer_tensor = torch.empty(length_tensor.item(), dtype=torch.uint8, device=device)
        # 广播实际内容
        dist.broadcast(buffer_tensor, src=0)
        # 从bytes反序列化
        if rank != 0:
            data = pickle.loads(buffer_tensor.cpu().numpy().tobytes())

        dist.barrier()
        return data