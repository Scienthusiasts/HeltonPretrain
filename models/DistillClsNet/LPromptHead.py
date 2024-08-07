import torch
import torch.nn as nn
import timm
from torchvision import models

from models.YOLOBlock import *
from utils.utils import *
from models import CLIP
from loss.loss import InfoNCELoss, TripleLoss, ThresholdMarginLoss, IdClassifyLoss



class LPromptHead(nn.Module):
    '''Backbone
    '''
    def __init__(self, cls_num:int, input_c:int, clip_embedding_c:int, kernel_s:list[int], mid_c:list[int], add_share_head=True):
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
        super(LPromptHead, self).__init__()
        self.cat_nums = cls_num
        self.clip_embedding_c = clip_embedding_c
        '''损失函数'''
        self.clsLoss = nn.CrossEntropyLoss()
        self.contrastLoss = InfoNCELoss()
        self.distillLoss = nn.SmoothL1Loss()
        # 针对可学习动态阈值设定的损失
        self.TMarginLoss = ThresholdMarginLoss()
        self.idClassifyLoss = IdClassifyLoss()

        # 特征提取
        if add_share_head:
            self.share_head = nn.Sequential(
                Conv(input_c , mid_c[0], kernel_s[0], 1, 0),
                Conv(mid_c[0], mid_c[1], kernel_s[1], 1, 0),
                Conv(mid_c[1], mid_c[2], kernel_s[2], 1, 0),
            )
            # 分类头
            self.cls_head = nn.Linear(mid_c[2], self.cat_nums)
            self.clip_head = nn.Linear(mid_c[2], clip_embedding_c)
        else:
            self.share_head = nn.Identity()
            self.cls_head = nn.Linear(input_c, self.cat_nums)
            self.clip_head = nn.Linear(input_c, clip_embedding_c)

        # 无论最后尺寸多大，都池化成1x1,这样输入的图像尺寸就可以是任意大小,但必须大于224x224
        self.gap = nn.AdaptiveAvgPool2d(1)
        '''可学习动态阈值'''
        # self.learnable_T = nn.Sequential(
        #     nn.BatchNorm1d(clip_embedding_c*2),
        #     nn.Linear(clip_embedding_c*2, 1),
        # )
        self.l_prompts = nn.Parameter(torch.zeros((cls_num, clip_embedding_c)))
        # 权重初始化
        init_weights(self.share_head, 'normal', 0, 0.01)
        init_weights(self.cls_head, 'normal', 0, 0.01)
        init_weights(self.clip_head, 'normal', 0, 0.01)
        # init_weights(self.learnable_T, 'normal', 0, 0.01)



    def forward(self, x:torch.tensor):
        '''前向传播
        '''
        x = self.share_head(x)
        x = self.gap(x)
        x = x.reshape(x.shape[0], -1)
        cls_logits = self.cls_head(x)
        embeddings = self.clip_head(x)
        return cls_logits, embeddings
    


