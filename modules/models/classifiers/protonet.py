import torch
import torch.nn as nn
import timm
import torch.distributed as dist
# 注册机制
from register import MODELS


@MODELS.register
class ProtoNet(nn.Module):
    """Prototype Network for Image Classification

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
            pred = self.head(feats[-1])
            return pred
        
        else:
            x, y = datas[0], datas[1]
            feats = self.backbone(x)  
            losses = self.head.loss(feats[-1], y)
            return losses



    def forward_with_protoheatmap(self, datas):
        """前向 + 可视化prototype的注意力热图
            Args:
                datas:       dataloader传来的图像+标签
            Returns:
                pred:        逐类别特征图的余弦相似度之和[B, nc]
                sim_heatmap: 余弦相似度特征图, 可以用来可视化
        """
        feats = self.backbone(datas)  
        pred, sim_heatmap = self.head.forward_with_protoheatmap(feats[-1])
        return pred, sim_heatmap
