import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.ckpts_utils import save_ckpt




class NecessaryHook():
    """实现训练/评估时一定会用到的hooks
    """
    def __init__(self, eval_pipeline):
        """
            Args:
                eval_pipeline: 任务特定的评估实例
        """
        self.eval_pipeline = eval_pipeline


    def hook_after_batch(self, runner):
        """batch级别日志 hook
            Args:
                runner: Runner实例
        """
        if runner.accelerator.is_main_process:
            # 记录/打印日志
            runner.runner_logger.train_iter_log_printer(runner.cur_step, runner.cur_epoch, runner.optimizer, runner.losses)


    def hook_after_epoch(self, runner):
        """epoch级别日志 + 保存权重 hook
            Args:
                runner: Runner实例
        """
        if (runner.cur_epoch % runner.eval_interval == 0 or runner.cur_epoch == runner.epoch) and runner.accelerator.is_main_process:
            # 评估+记录/打印日志
            flag_metric_name = self.hook_after_eval(runner)
            # 保存权重
            if runner.accelerator.is_main_process:
                model_unwrapped = runner.accelerator.unwrap_model(runner.model)
                save_ckpt(runner.cur_epoch, runner.eval_interval, model_unwrapped, runner.scheduler,
                        runner.log_dir, runner.runner_logger.argsHistory, flag_metric_name)

    def hook_after_eval(self, runner):
        """评估时 hook
            Args:
                runner: Runner实例
        """
        # 需要解包构建一个非DDP包装的模型副本，否则如果只用gpu0上的模型推理时, 
        # 一些操作会使用跨进程通信(allreduce / broadcast), 此时会产生阻塞
        model_unwrapped = runner.accelerator.unwrap_model(runner.model)
        # 评估
        evaluations, flag_metric_name = self.eval_pipeline(runner, model_unwrapped)
        # 记录/打印日志
        runner.runner_logger.train_epoch_log_printer(runner.cur_epoch, evaluations, flag_metric_name)
        return flag_metric_name