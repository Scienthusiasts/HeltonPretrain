import torch
import torch.nn as nn
import timm
import torch.distributed as dist
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix
# 注册机制
from heltonx.utils.register import MODELS


@MODELS.register
class MLPNetMultiTasks(nn.Module):
    """
    """
    def __init__(self, backbone:nn.Module, cls_head:nn.Module, emb_head:nn.Module, load_ckpt=None, ensemble_pred=False):
        """
        Args:
            backbone:  网络的骨干部分
            head:      网络的头部
            load_ckpt: 是否加载本地预训练权重(自定义权重)
        """
        super().__init__()
        self.ensemble_pred = ensemble_pred
        # 网络模块
        self.backbone = backbone
        self.cls_head = cls_head
        self.emb_head = emb_head
        # 是否导入预训练权重
        if load_ckpt: 
            self = load_state_dict_with_prefix(self, load_ckpt, ['model.'])


    def forward(self, datas, return_loss=False):
        """前向 / 计算损失(注意这里一定得把loss合并到forward中, 否则只调用loss方法得绕过module, DDP合并不了不同gpu的梯度)
        Args:  
            datas:       return_loss=True时是dataloader传来的图像+标签, 否则只有图像
            return_loss: True则计算损失, 返回损失
        Returns:
            pred: 模型预测结果(logits, 未经过softmax)
            loss: 模型损失
        """
        if not return_loss:
            # feats是多尺度特征图
            feats = self.backbone(datas)  
            cls_logits = self.cls_head(feats[-1])
            if self.ensemble_pred:
                emb_logits = self.emb_head(feats[-1])
                # TODO:这里还有问题, 两个logits的分布可能不一致, 不能单纯加权
                return 0.5 * (cls_logits + emb_logits)
            else:
                return cls_logits
        else:
            x, y = datas[0], datas[1]
            feats = self.backbone(x)  
            cls_losses = self.cls_head.loss(feats[-1], y)
            emb_losses = self.emb_head.loss(feats[-1], y, x)
            # 合并两个损失(如果键重复, 后面的字典会覆盖前面的值)
            losses = cls_losses | emb_losses
            return losses

