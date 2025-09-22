import inspect
import torch.optim.lr_scheduler as lr_scheduler
from register import SCHEDULERS

# 批量注册 torch.optim.lr_scheduler 里的常见scheduler
# 遍历 lr_scheduler 模块下的所有类
# ['ChainedScheduler', 'ConstantLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 
# 'CyclicLR', 'ExponentialLR', 'LRScheduler', 'LambdaLR', 'LinearLR', 'MultiStepLR', 'MultiplicativeLR', 
# 'OneCycleLR', 'PolynomialLR', 'ReduceLROnPlateau', 'SequentialLR', 'StepLR']
for name, obj in inspect.getmembers(lr_scheduler, inspect.isclass):
    # 排除私有类（以 _ 开头的）
    if name.startswith("_"):
        continue
    # 过滤出属于 lr_scheduler 模块的类（排除继承链里导入的外部类）
    if obj.__module__ == lr_scheduler.__name__:
        SCHEDULERS.add_item(name, obj)


@SCHEDULERS.register
class WarmupScheduler:
    """带warmup的通用Scheduler
    """
    def __init__(self, base_scheduler, optimizer, warmup_epochs=5, min_lr=0.0, last_epoch=0):
        """
            base_scheduler: 已经实例化的base scheduler(WarmupScheduler套壳在这上面)
            optimizer:      已经实例化的优化器
            warmup_epochs:  多少iter后结束warmup(结束后的学习率就为optimizer的学习率)
            min_lr:         warmup一开始的学习率
        """
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        
        # base_scheduler 的初始 lr 作为 warmup 的目标 lr
        self.target_lrs = [group['lr'] for group in optimizer.param_groups]

        # warmup 起始 lr 设置为 min_lr
        for group in optimizer.param_groups:
            group['lr'] = self.min_lr

        self.last_epoch = last_epoch

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 0-based warmup，Epoch 1 lr = min_lr
            warmup_factor = float(self.last_epoch) / float(self.warmup_epochs)
            return [self.min_lr + (target_lr - self.min_lr) * warmup_factor
                    for target_lr in self.target_lrs]
        else:
            return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            # warmup 阶段手动更新 optimizer.lr
            lr_list = self.get_lr()
            for idx, group in enumerate(self.optimizer.param_groups):
                group['lr'] = lr_list[idx]
        else:
            # warmup 结束后调用 base_scheduler.step()
            if epoch is not None:
                self.base_scheduler.step(epoch - self.warmup_epochs)
            else:
                self.base_scheduler.step()
        self.last_epoch += 1

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]












if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim

    base_schedulers_cfgs = {
        "type": "StepLR",
        # 每间隔step_size个iter更新学习率
        "step_size": 1000,
        # 每次学习率变为原来的gamma倍
        "gamma": 0.1,
    }
    warmup_schedulers_cfgs = {
        "type": "WarmupScheduler",
        "min_lr": 1e-6,
        "warmup_epochs": 2000
    }
    # 假设一个简单的模型
    model = nn.Linear(256, 2)

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    base_scheduler = SCHEDULERS.build_from_cfg(base_schedulers_cfgs, optimizer=optimizer)
    scheduler = SCHEDULERS.build_from_cfg(warmup_schedulers_cfgs, base_scheduler=base_scheduler, optimizer=optimizer)

    # 数据 (只是示例, 随便造点数据)
    x = torch.randn(1000, 256)
    y = torch.randint(0, 2, (1000,))

    criterion = nn.CrossEntropyLoss()

    for epoch in range(20000):  # 共训练 20 个 epoch
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # 每个 epoch 结束后更新学习率
        if epoch % 50 == 0:
            scheduler.step()

        # 打印当前学习率
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}, Loss={loss.item():.4f}, LR={lr:.6f}")