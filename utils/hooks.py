import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils import *
from utils.eval_utils import eval_epoch





def hook_after_batch(runner):
    """batch级别日志 hook
        Args:
            runner: Runner实例
    """
    if runner.mode == 'train' or (runner.mode == 'train_ddp' and dist.get_rank() == 0):
        # 记录/打印日志
        runner.runner_logger.train_iter_log_printer(runner.cur_step, runner.cur_epoch, runner.optimizer, runner.losses)



def hook_after_epoch(runner):
    """epoch级别日志 + 保存权重 hook
        Args:
            runner: Runner实例
    """
    if runner.mode == 'train' or (runner.mode == 'train_ddp' and dist.get_rank() == 0):
        if runner.cur_epoch % runner.eval_interval == 0:
            # 评估
            evaluations, flag_metric_name = eval_epoch(
                runner.device, None, runner.model, runner.valid_dataloader,
                runner.train_dataset.cat_names, runner.log_dir
            )
            # 记录/打印日志
            runner.runner_logger.train_epoch_log_printer(runner.cur_epoch, evaluations, flag_metric_name)
            # 保存权重
            save_ckpt(runner.cur_epoch, runner.eval_interval, runner.model, runner.optimizer,
                      runner.log_dir, runner.runner_logger.argsHistory, flag_metric_name)
