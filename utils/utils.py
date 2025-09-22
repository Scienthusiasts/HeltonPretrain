import torch.nn as nn
import random
import torch
import numpy as np
import os



def init_weights(model, init_type, mean=0, std=0.01):
    '''权重初始化方法
    '''
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if init_type=='he':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if init_type=='normal':
                # 使用高斯随机初始化
                nn.init.normal_(module.weight, mean=mean, std=std)  
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


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



def save_ckpt(epoch, eval_interval, model, optimizer, log_dir, argsHistory, flag_metric_name):
    '''保存权重和训练断点
        Args:
            - epoch:       当前epoch
            - model:       网络模型实例
            - optimizer:   优化器实例
            - log_dir:     日志文件保存目录
            - argsHistory: 日志文件记录实例
            - logger:      日志输出实例

        Returns:
            None
    '''  
    # checkpoint_dict能够恢复断点训练
    checkpoint_dict = {
        'epoch': epoch, 
        'model_state_dict': model.state_dict(), 
        'optim_state_dict': optimizer.state_dict()
        }
    torch.save(checkpoint_dict, os.path.join(log_dir, f"train_epoch{epoch}.pt"))
    torch.save(model.state_dict(), os.path.join(log_dir, "last.pt"))
    # 如果本次Epoch的参考指标最大，则保存网络参数
    flag_metric_list = argsHistory.args_history_dict[flag_metric_name]
    best_flag_metric_val = max(flag_metric_list)
    best_epoch = flag_metric_list.index(best_flag_metric_val) + 1
    if epoch == best_epoch * eval_interval:
        torch.save(model.state_dict(), os.path.join(log_dir, f'best_{flag_metric_name}.pt'))