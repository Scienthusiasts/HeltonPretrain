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








class JSDLoss(nn.Module):
    '''计算两个概率分布的Jensen-Shannon Divergence, 可以在蒸馏时代替SmoothL1试试
       p=[bs, C], q=[bs, C]
    '''
    def __init__(self):
        super(JSDLoss, self).__init__()
        
    def forward(self, p, q):
        m = 0.5 * (p + q)
        kl_pm = F.kl_div(p.log(), m)
        kl_qm = F.kl_div(q.log(), m)
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd.mean()








class SoftTripleLoss(nn.Module):
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
    





class ThresholdMarginLoss(nn.Module):
    '''Threshold Margin Loss: 针对动态可学习阈值的margin损失, 使得对于正样本对, 可学习阈值低于这两个样本的相似度; 对于负样本对, 可学习阈值高于这两个样本的相似度

    '''
    def __init__(self, alpha=1., gamma=1.):
        super(ThresholdMarginLoss, self,).__init__()
        # margin的超参
        self.alpha = alpha
        # 平衡正负样本损失权重的超参(正负样本数量不均衡)
        self.gamma = gamma
        
    def forward_(self, T_l, x, x_aug, scale=100.):
        '''正负样本不平衡采样'''
        norm_T_l = F.sigmoid(T_l)
        norm_simM = COSSim(x, x_aug) / scale
        '''计算负样本的margin'''
        margin = norm_T_l - norm_simM
        '''正样本(对角线)的margin符号取反'''
        # 获取对角线元素的索引
        indices = torch.arange(T_l.shape[0])
        # 将对角线元素的符号取反
        margin[indices, indices] = -margin[indices, indices]
        weightM = torch.eye(T_l.shape[0]).to(x.device) * (T_l.shape[0]-1) * self.gamma + 1
        # print(torch.sum(margin>0).item() / (T_l.shape[0]*T_l.shape[0]))
        loss = torch.exp(-self.alpha * margin) * weightM 
        return loss.mean()

    def forward(self, T_l, x, x_aug, scale=100.):
        '''正负样本平衡采样'''
        norm_T_l = F.sigmoid(T_l)
        norm_simM = COSSim(x, x_aug) / scale
        '''计算负样本的margin'''
        margin = norm_T_l - norm_simM
        '''正样本(对角线)的margin符号取反'''
        # 获取对角线元素的索引
        indices = torch.arange(T_l.shape[0])
        # 将对角线元素的符号取反
        margin[indices, indices] = -margin[indices, indices]
        '''获取正样本'''
        pos_samples = margin[indices, indices]
        '''随机采样负样本,个数等于正样本个数'''
        # 创建一个 bool 掩码，标记对角线元素
        mask = torch.eye(T_l.shape[0], dtype=bool)
        # 获取非对角线元素的索引
        non_diag_indices = torch.nonzero(~mask, as_tuple=False)
        # 随机采样 n 个非对角线元素
        neg_indices = non_diag_indices[torch.randperm(non_diag_indices.size(0))[:T_l.shape[0]]]
        # 根据采样的索引从 A 中提取对应的元素
        neg_samples = margin[neg_indices[:, 0], neg_indices[:, 1]]
        # 将正负样本拼在一起
        samples = torch.cat((pos_samples, neg_samples), dim=0)
        print(torch.sum(samples>0).item() / (T_l.shape[0]*2))
        loss = torch.exp(-self.alpha * samples)
        return loss.mean()
    






class IdClassifyLoss(nn.Module):
    def __init__(self, ):
        super(IdClassifyLoss, self,).__init__()
        self.loss = nn.CrossEntropyLoss()


    def forward(self, learnable_T, target_GT):
        '''获取正样本(对角线)'''
        # print(learnable_T.shape, target_GT.shape)
        pos_samples = learnable_T[target_GT==1]
        pos_labels = target_GT[target_GT==1]
        '''随机采样负样本,个数等于正样本个数'''
        # 获取正样本索引
        non_diag_indices = torch.nonzero(target_GT == 0, as_tuple=True)[0]
        # 随机采样 n 个负样本索引
        neg_indices = non_diag_indices[torch.randperm(non_diag_indices.size(0))[:pos_samples.shape[0]]]
        # 根据采样的索引从 A 中提取对应的元素
        neg_samples = learnable_T[neg_indices]
        neg_labels = target_GT[neg_indices]
        # 将正负样本拼在一起
        samples = torch.cat((pos_samples, neg_samples), dim=0)
        labels = torch.cat((pos_labels, neg_labels), dim=0)
        print(samples.shape, labels.shape)
        print(torch.sum(torch.argmax(samples, dim=1)==labels)/samples.shape[0])
        # print(torch.sum(samples>0).item() / samples.shape[0])
        loss = self.loss(samples, labels)
        return loss