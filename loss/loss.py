import torch.nn as nn
import torch
import numpy as np

from utils.clipUtils import COSSim






class InfoNCELoss(nn.Module):
    '''InfoNCELoss
    '''
    # backbone_name no used, but must kept
    def __init__(self):
        super(InfoNCELoss, self).__init__()
        # NOTE:self.scale暂时设置为不可学习的，否则会报错 RuntimeError: Trying to backward through the graph a second time
        # self.scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp().detach()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, vec1, vec2, label):
        # 先计算两个向量相似度
        sim_logits = COSSim(vec1, vec2)
        # 相似度结果作为logits计算交叉熵损失
        loss = self.loss(sim_logits, label)
        return loss
