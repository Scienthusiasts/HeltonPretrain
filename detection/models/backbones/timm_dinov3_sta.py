import torch.nn as nn
import timm
import torch.nn.functional as F
from heltonx.utils.ckpts_utils import *
from heltonx.utils.utils import init_weights
# 注册机制
from heltonx.utils.register import MODELS
from heltonx.utils.utils import multi_apply







class ConvBlock(nn.Module):
    """Conv + BN + ReLU 基本卷积块
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)




class BiFusion(nn.Module):
    """
    Bi-Fusion Module
        Args:
            A: [B, C1, H1, W1] (context, e.g. semantic feature)
            B: [B, C2, H2, W2] (details, e.g. low-level feature)
        Returns:
            out: [B, C, H2, W2]
    """
    def __init__(self, A_ch, B_ch, out_ch):
        super(BiFusion, self).__init__()
        # 融合卷积
        self.fuse_conv = ConvBlock(A_ch + B_ch, out_ch, 1, 1, 0)


    def forward(self, A, B):
        # 1. 将A上采样到B的空间尺寸
        A_upsampled = F.interpolate(A, size=B.shape[2:], mode='bilinear', align_corners=False)
        # 2. 拼接通道
        fused = torch.cat([A_upsampled, B], dim=1)
        # 3. 1x1卷积融合
        out = self.fuse_conv(fused)
        return out
    



class SpatialTuningAdapter(nn.Module):
    """
    Spatial Tuning Adapter (STA)
    输入:  原始图像或浅层特征
    输出:  多尺度特征 [1/8, 1/16, 1/32]
    """
    def __init__(self, in_ch=3, layer_dims=[128, 256, 512, 1024, 2048]):
        super(SpatialTuningAdapter, self).__init__()
        self.base_conv = ConvBlock(in_ch, layer_dims[0], k=3, s=2, p=1)
        layers = [ConvBlock(layer_dims[i], layer_dims[i+1], k=3, s=2, p=1) for i in range(4)]
        self.layers = nn.ModuleList(layers)


    def forward(self, x):
        x = self.base_conv(x)
        lvl_feats = []
        for layer in self.layers:
            x = layer(x)
            lvl_feats.append(x)

        return lvl_feats[1:]





@MODELS.register
class DINOv3STA(nn.Module):
    """DINOv3STA DINOv3+多尺度adapter
    """
    def __init__(self, dino_name: str, sta_layer_dims, fuse_layer_dims, dino_ckpt, froze_dino=True):
        """
        Args:
            dino_name:       timm 中dino模型名称
            sta_layer_dims:  指定sta每一层的维度
            fuse_layer_dims: 每一层bi_fusion后的维度
            froze_dino:      是否冻结dino骨干网络
            dino_ckpt:       加载dino预训练权重
        """
        super().__init__()
        # features_only=True 直接去掉分类头
        self.dinov3 = timm.create_model(dino_name, pretrained=False, features_only=True, out_indices=[5, 8, 11])
        self.sta = SpatialTuningAdapter(3, sta_layer_dims)
        self.bi_fusions = nn.ModuleList([BiFusion(384, sta_layer_dims[2+i], fuse_layer_dims[i]) for i in range(3)])
        # 导入dino权重
        self.dinov3 = load_state_dict_with_prefix(self.dinov3, dino_ckpt, prefixes_to_try=['model.'])
        # 是否冻结dinov3权重
        if froze_dino:
            for param in self.dinov3.parameters():
                param.requires_grad_(False)
        # 初始化
        for m in self.sta.modules():
            init_weights(m, 'normal', 0, 0.01)
        for m in self.bi_fusions.modules():
            init_weights(m, 'normal', 0, 0.01)



    def forward_single(self, dino_feat, sta_feat, i):
        """单层融合前向
        """
        lvl_x = self.bi_fusions[i](dino_feat, sta_feat)
        return lvl_x


    def forward(self, x):
        """前向传播
        """
        dino_x = self.dinov3(x)
        sta_x = self.sta(x)
        n = range(len(sta_x))
        lvl_feats = multi_apply(self.forward_single, dino_x, sta_x, n)
        return lvl_feats





# for test only:
if __name__ == '__main__':

    # 配置字典
    cfgs=dict(
        type="DINOv3STA",
        dino_name="vit_small_patch16_dinov3.lvd1689m",
        sta_layer_dims=[64, 128, 256, 512, 1024],
        fuse_layer_dims=[256, 512, 1024],
        dino_ckpt="ckpts/vit_small_patch16_dinov3.lvd1689m.pt",
        froze_dino=True
    )
    backbone = MODELS.build_from_cfg(cfgs)

    x = torch.randn(4, 3, 640, 640)
    x = backbone(x)

    for o in x:
        print(o.shape)



