import torch.nn as nn
import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
import math

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







class SoftTriple(nn.Module):
    '''SoftTriple 损失: 无需三元组采样的深度度量学习
       https://github.com/idstcv/SoftTriple/blob/master/loss/SoftTriple.py

    Args:
        - la:  Eq(8) lambda
        - gamma:
        - tau:
        - margin: 
        - dim:    embedding 的维度
        - cN:     数据集类别数
        - K:      类别聚类中心数

    Return:
    
    '''
    def __init__(self, dim, cN, K=10, la=20, gamma=0.1, tau=0.2, margin=0.01):
        super(SoftTriple, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN*K))
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        # 对fc的weight进行normalize
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify
        






class TripleLoss(nn.Module):
    '''Triple Loss: 三元组损失

    '''
    def __init__(self, margin, p):
        super(TripleLoss, self,).__init__()
        self.loss = nn.TripletMarginLoss(margin=margin, p=p)
    def forward(self, x, contrast_x):
        # 生成一个随机排列的索引
        shuffle_idx = torch.randperm(x.shape[0])
        anchor = x
        pos = contrast_x
        neg = contrast_x[shuffle_idx]
        loss = self.loss(anchor, pos, neg)
        return loss
    
