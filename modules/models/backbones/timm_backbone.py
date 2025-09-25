import torch
import torch.nn as nn
import timm
import torch.distributed as dist
# 注册机制
from register import MODELS


@MODELS.register
class TIMMBackbone(nn.Module):
    """通用 TIMM Backbone 模块

    """
    def __init__(self, model_name: str, pretrained=True, out_layers=None, froze_backbone=False, load_ckpt=None):
        """
        Args:
            model_name:     str, timm 中模型名称
            pretrained:     是否加载预训练权重(官方的权重)
            features_only:  是否去掉分类头，仅保留特征提取层
            out_layers:     list[int] 或 None, 指定哪些 stage 层输出特征
            froze_backbone: 是否冻结骨干网络
            load_ckpt:      是否加载本地预训练权重(自定义权重)
        """
        super().__init__()
        # features_only=True 直接去掉分类头
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=out_layers)
        # 是否冻结backbone
        if froze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

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
        # 返回一个列表
        x = self.backbone(x)  
        return x



if __name__ == '__main__':
    # 配置字典
    cfg = {
        "type": "TIMMBackbone",
        "model_name": "vit_small_patch16_dinov3.lvd1689m",
        "pretrained": r"/mnt/yht/code/HeltonPretrain/ckpts/vit_small_patch16_dinov3.lvd1689m.pt",
        "out_layers": [11],
        "froze_backbone": True,
        "load_ckpt": None

    }
    backbone = MODELS.build_from_cfg(cfg)
    x = torch.randn(2, 3, 256, 256)
    features = backbone(x)
    for i, f in enumerate(features):
        print(f"stage {i} output shape: {f.shape}")