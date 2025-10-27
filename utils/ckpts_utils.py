# 负责权重load 和save相关逻辑
import torch.nn as nn
import torch
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist












def load_state_dict_with_prefix(model, load_ckpt, prefixes_to_try=['model.', 'module.', 'encoder.', 'backbone.', 'teacher.', 'student.'], state_dict=None):
    """自动处理权重键名前缀不匹配问题（双向适配）
    Args:
        model: 要加载权重的模型
        load_ckpt: 权重文件路径
        prefixes_to_try: 要尝试的前缀列表，默认包含常见的训练保存前缀
    Returns:
        加载了权重的模型
    """
    use_ddp = dist.is_initialized()
    if not state_dict:
        if not use_ddp or use_ddp and dist.get_rank() == 0:
            print(f"➡️  loadong ckpt: {load_ckpt}")

        state_dict = torch.load(load_ckpt, map_location='cpu')
    
    # 首先提取模型权重（处理checkpoint中可能包含的其他信息）
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    model_state_dict = model.state_dict()

    if not use_ddp or use_ddp and dist.get_rank() == 0:
        print(f"模型键数量: {len(model_state_dict)}, 权重键数量: {len(state_dict)}")
    
    # 尝试不同的前缀匹配策略
    best_match_ratio = 0
    best_state_dict = None
    best_strategy = "原始匹配"
    
    # 策略1: 原始匹配（不处理前缀）
    matching_keys = set(model_state_dict.keys()) & set(state_dict.keys())
    match_ratio = len(matching_keys) / len(model_state_dict) if model_state_dict else 0
    if match_ratio > best_match_ratio:
        best_match_ratio = match_ratio
        best_state_dict = state_dict
        best_strategy = "原始匹配"
    
    # 策略2: 去除权重中的前缀（权重比模型多前缀）
    for prefix in prefixes_to_try:
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        matching_keys = set(model_state_dict.keys()) & set(new_state_dict.keys())
        match_ratio = len(matching_keys) / len(model_state_dict) if model_state_dict else 0
        
        if match_ratio > best_match_ratio:
            best_match_ratio = match_ratio
            best_state_dict = new_state_dict
            best_strategy = f"去除权重前缀 '{prefix}'"
    
    # 策略3: 为权重添加前缀（模型比权重多前缀）
    for prefix in prefixes_to_try:
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = prefix + key
            new_state_dict[new_key] = value
        
        matching_keys = set(model_state_dict.keys()) & set(new_state_dict.keys())
        match_ratio = len(matching_keys) / len(model_state_dict) if model_state_dict else 0
        
        if match_ratio > best_match_ratio:
            best_match_ratio = match_ratio
            best_state_dict = new_state_dict
            best_strategy = f"添加权重前缀 '{prefix}'"
    
    if not use_ddp or use_ddp and dist.get_rank() == 0:
        print(f"最佳匹配策略: {best_strategy}, 匹配度: {best_match_ratio:.1%}")
    
    # 使用最佳匹配策略加载权重
    missing_keys, unexpected_keys = model.load_state_dict(best_state_dict, strict=False)
    
    # 详细输出匹配情况
    if not use_ddp or use_ddp and dist.get_rank() == 0:
        if missing_keys:
            print(f"⚠️  缺失的键 ({len(missing_keys)}个):")
            for key in missing_keys[:5]:  # 只显示前5个
                print(f"   - {key}")
            if len(missing_keys) > 5:
                print(f"   ... 还有 {len(missing_keys) - 5} 个")
        
        if unexpected_keys:
            print(f"⚠️  多余的键 ({len(unexpected_keys)}个):")
            for key in unexpected_keys[:5]:  # 只显示前5个
                print(f"   - {key}")
            if len(unexpected_keys) > 5:
                print(f"   ... 还有 {len(unexpected_keys) - 5} 个")
        
        print(f"✅ 权重加载完成 - 匹配度: {best_match_ratio:.1%}")
    
    return model










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
    # resume后开始的epoch
    resume_epoch = ckpt['epoch'] + 1
    scheduler.last_epoch = batch_nums * resume_epoch
    # model.load_state_dict(ckpt['model_state_dict'])
    model = load_state_dict_with_prefix(model, load_ckpt=None, state_dict=ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optim_state_dict'])
    scheduler.base_scheduler.load_state_dict(ckpt['sched_state_dict'])

    # 主节点才进行日志记录
    use_ddp = dist.is_initialized()
    if not use_ddp or use_ddp and dist.get_rank() == 0:
        runner_logger.logger.info(f'resume:{resume}')
        runner_logger.logger.info(f'resume_epoch:{resume_epoch}')
        # 导入上一次中断训练时的args
        json_dir, _ = os.path.split(resume)
        runner_logger.argsHistory.loadRecord(json_dir)

    return resume_epoch

