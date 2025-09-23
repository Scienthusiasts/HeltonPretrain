import torch
import torch.nn as nn
import timm
import torch.distributed as dist
# 注册机制
from register import MODELS


@MODELS.register
class FCNet(nn.Module):
    """Multi-scale Feature Fusion Network 多尺度特征融合的图像分类模型

    """
    def __init__(self, backbone:nn.Module, head:nn.Module, load_ckpt=None):
        """
        Args:
            backbone:  网络的骨干部分
            head:      网络的头部
            load_ckpt: 是否加载本地预训练权重(自定义权重)
        """
        super().__init__()
        # 网络模块
        self.backbone = backbone
        # 无论最后尺寸多大，都池化成1x1,这样输入的图像尺寸就可以是任意大小,但必须大于224x224
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = head
        # 是否导入预训练权重
        if load_ckpt: 
            # self.load_state_dict(torch.load(load_ckpt, map_location='cuda:{}'.format(dist.get_rank())))
            self.load_state_dict(torch.load(load_ckpt))


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
            # 取最后一层特征池化后接MLP层
            last_feat = self.gap(feats[-1]).flatten(1)
            pred = self.head(last_feat)
            return pred
        
        else:
            x, y = datas[0], datas[1]
            feats = self.backbone(x)  
            last_feat = self.gap(feats[-1]).flatten(1)
            losses = self.head.loss(last_feat, y)
            return losses

