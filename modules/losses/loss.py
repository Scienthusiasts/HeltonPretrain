import torch
import torch.nn as nn
from register import MODELS



@MODELS.register
class CELoss(nn.Module):
    '''CELoss 多分类交叉熵损失(适合输出的概率总和=1, 互斥的多分类任务)
    '''
    def __init__(self, reduction='mean'):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [B, cls_num] 
            labels: [B, ]
        """
        loss = self.loss(logits, labels)
        return loss
    
    

@MODELS.register
class MultiClassBCELoss(nn.Module):
    '''MultiClassBCELoss 使用BCELoss实现多分类任务(非互斥的多分类任务)
    '''
    def __init__(self, reduction='mean'):
        super(MultiClassBCELoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(self, scores, labels):
        """
        Args:
            scores: [B, cls_num]
            labels: [B, ]
        """
        # 将类别索引转换为one-hot编码 [B, ] -> [B, cls_num]
        onehot_labels = torch.zeros_like(scores)
        onehot_labels.scatter_(1, labels.unsqueeze(1), 1.0) 
        loss = self.loss(scores, onehot_labels)
        return loss
    


@MODELS.register
class SmoothL1Loss(nn.Module):
    '''SmoothL1Loss
    '''
    def __init__(self, reduction='mean'):
        super(SmoothL1Loss, self).__init__()
        self.loss = nn.SmoothL1Loss(reduction=reduction)
    
    def forward(self, x, y):
        """
        Args:
            x: [B, D] 
            y: [B, D]
        """
        loss = self.loss(x, y)
        return loss



@MODELS.register
class KLDivLoss(nn.Module):
    '''KL散度 损失
    '''
    def __init__(self, dist_dim, reduction='mean'):
        super(KLDivLoss, self).__init__()
        self.KLDivLoss = nn.KLDivLoss(reduction='none')
        self.dist_dim = dist_dim
        self.reduction = reduction

    def forward(self, s, t, to_distribution=True, loss_weight=1.):
        '''
            s:               输入分布1, 大致均值0方差1, 形状为 [bs, n, dim] (其中一个维度表示分布的维度, 另一个维度表示有几个分布)
            t:               输入分布2, 大致均值0方差1, 形状为 [bs, n, dim] (其中一个维度表示分布的维度, 另一个维度表示有几个分布)
            to_distribution: 是否将输入归一化为概率分布
            dist_dim:        表示哪一个维度表示分布的维度(默认1), 1或2
        '''
        if to_distribution:
            # 将s和t第一维度转化为频率(和=1)
            # 计算 s 的对数概率 (KLDivLoss的input要求)
            # TODO: softmax是否要加温度系数
            log_s = nn.LogSoftmax(dim=self.dist_dim)(s)
            t = torch.softmax(t, dim=self.dist_dim)

        # 调整输入形状，确保 nn.KLDivLoss 对分布维度计算 KLD(如果分布维度是 2, 则已经是最后一维无需转置)
        if self.dist_dim == 1:
            # 如果分布维度不是最后一维, 把分布维度调整到最后一维
            log_s = log_s.transpose(1, 2)
            t = t.transpose(1, 2)

        # 注意: nn.KLDivLoss 默认认为分布的维度是最后一个维度
        kld_loss = self.KLDivLoss(log_s, t) 
        if self.reduction=='mean':
            return kld_loss.sum(dim=-1).mean() * loss_weight
        if self.reduction=='sum':
            return kld_loss.sum() * loss_weight
        if self.reduction=='none':
            return kld_loss.mean(dim=-1)
        



@MODELS.register
class JSDivLoss(nn.Module):
    '''Jensen-Shannon散度 损失
    '''
    def __init__(self, dist_dim, reduction='mean'):
        super(JSDivLoss, self).__init__()
        self.KLDivLoss = nn.KLDivLoss(reduction='none')
        self.dist_dim = dist_dim
        self.reduction = reduction
        self.e = 1e-8

    def forward(self, s, t, to_distribution=True, loss_weight=1.):
        '''
            s:               输入分布 1, 形状为 [n,m] (其中一个维度表示分布的维度, 另一个维度表示有几个分布)
            t:               输入分布 2, 形状为 [n,m] (其中一个维度表示分布的维度, 另一个维度表示有几个分布)
            to_distribution: 是否将输入归一化为概率分布
            dist_dim:        表示哪一个维度表示分布的维度(默认0)
        '''
        # 将s和t第一维度转化为频率(和=1)
        if to_distribution:
            # 在归一化后加 self.e，然后再重新归一化, 避免直接加 self.e 可能会导致分布的变化
            # 将 s 和 t 归一化为概率分布(.sum一般不可能是0, 所以没加e)
            s = s / s.sum(dim=self.dist_dim, keepdim=True)
            t = t / t.sum(dim=self.dist_dim, keepdim=True)
            # 加入小常数，避免零值
            s = s + self.e
            t = t + self.e
            # 重新归一化
            s = s / s.sum(dim=self.dist_dim, keepdim=True)
            t = t / t.sum(dim=self.dist_dim, keepdim=True)
        # 计算平均分布的对数
        log_mean = ((s + t) * 0.5 + self.e).log()

        # 调整输入形状，确保 nn.KLDivLoss 对分布维度计算 KLD(如果分布维度是 1, 则无需转置)
        if self.dist_dim == 0:
            # 如果分布维度是0, 把分布维度调整到1
            log_mean = log_mean.transpose(0, 1)
            s = s.transpose(0, 1)
            t = t.transpose(0, 1)

        # 注意: nn.KLDivLoss 默认认为分布的维度是最后一个维度
        jsd_loss = (self.KLDivLoss(log_mean, s) + self.KLDivLoss(log_mean, t)) * 0.5
        if self.reduction=='mean':
            return jsd_loss.mean() * loss_weight
        if self.reduction=='sum':
            return jsd_loss.sum() * loss_weight
        if self.reduction=='none':
            return jsd_loss.mean(dim=1)
