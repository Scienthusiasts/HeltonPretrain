import torch
import torch.nn as nn
import timm
from torchvision import models

from models.YOLOBlock import *
from utils.utils import *






class Head(nn.Module):
    '''Backbone
    '''
    def __init__(self, cls_num:int, input_c:int, kernel_s:list[int], mid_c:list[int], add_share_head=True):
        '''网络初始化

        Args:
            - backbone_name:  数据集类别数
            - cls_num:        使用哪个模型(timm库里的模型)
            - input_c:        使用哪个模型(timm库里的模型)
            - kernel_s:       sharehead里每个卷积核大小
            - mid_c:          sharehead里每个卷积通道数大小
            - add_share_head: 是否使用sharehead进一步提取特征
 
        Returns:
            None
        '''
        super(Head, self).__init__()
        '''损失函数'''
        self.clsLoss = nn.CrossEntropyLoss()
        '''网络组件'''
        # 包括背景类别
        self.cat_nums = cls_num
        # 特征提取
        if add_share_head:
            self.share_head = nn.Sequential(
                Conv(input_c , mid_c[0], kernel_s[0], 1, 0),
                Conv(mid_c[0], mid_c[1], kernel_s[1], 1, 0),
                Conv(mid_c[1], mid_c[2], kernel_s[2], 1, 0),
            )
            # 分类头
            self.cls = nn.Linear(mid_c[2], self.cat_nums)
        else:
            self.share_head = nn.Identity()
            self.cls = nn.Linear(input_c, self.cat_nums)

        # 无论最后尺寸多大，都池化成1x1,这样输入的图像尺寸就可以是任意大小,但必须大于224x224
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 权重初始化
        init_weights(self.share_head, 'normal', 0, 0.01)
        init_weights(self.cls, 'normal', 0, 0.01)



    def forward(self, x:torch.tensor):
        '''前向传播
        '''
        x = self.share_head(x)
        x = self.gap(x)
        x = x.reshape(x.shape[0], -1)
        out = self.cls(x)
        return out
    



    def batchLoss(self, x, batch_labels):
        pred = self.forward(x)
        cls_loss = self.clsLoss(pred, batch_labels)

        loss = dict(
            total_loss = cls_loss,
            cls_loss = cls_loss
        )

        return loss









# for test only:
if __name__ == '__main__':
    from models.Backbone import Backbone

    cls_num = 20
    backbone_name='cspresnext50.ra_in1k'
    mid_c = [512, 256, 256]
    backbone = Backbone(backbone_name, pretrain=False)
    head = Head(backbone_name, cls_num, mid_c)

    # 验证 
    x = torch.rand((4, 3, 224, 224))
    backbone_feat = backbone(x)
    out = head(backbone_feat)
    print(out.shape)