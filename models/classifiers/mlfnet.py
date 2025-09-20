import torch
import torch.nn as nn
import timm
import torch.distributed as dist
# 注册机制
from register import MODELS


@MODELS.register
class MLFNet(nn.Module):
    """多尺度融合的图像分类模型

    """
    def __init__(self, backbone:nn.Module, head:nn.Module, load_ckpt=None):
        """
        Args:
            backbone
            head
            load_ckpt:      是否加载本地预训练权重(自定义权重)
        """
        super().__init__()
        # 网络模块
        self.backbone = backbone
        self.head = head
        # 无论最后尺寸多大，都池化成1x1,这样输入的图像尺寸就可以是任意大小,但必须大于224x224
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 是否导入预训练权重
        if load_ckpt: 
            # self.load_state_dict(torch.load(load_ckpt, map_location='cuda:{}'.format(dist.get_rank())))
            self.load_state_dict(torch.load(load_ckpt))

    def forward(self, x):
        """
        Args:
            x: 输入图像张量 [B, C, H, W]
        Returns:
            输出特征列表，按 out_indices 顺序
        """
        # feats是多尺度特征图
        feats = self.backbone(x)  
        # 将不同尺度特征图池化后拼在一起
        cat_feats = torch.cat([self.gap(x).squeeze(2,3) for x in feats], dim=1)
        # 拼在一起的特征再过head
        fuse_feat = self.head(cat_feats)
        return fuse_feat

