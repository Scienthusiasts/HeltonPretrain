import torch
import torch.nn as nn
from utils.register import MODELS





@MODELS.register
class MSELoss(nn.Module):
    '''L2损失
    '''
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')
        self.reduction = reduction

    def forward(self, pred, target):
        """
        """
        loss = self.loss(pred, target)
        if self.reduction=='mean':
            return loss.mean()
        if self.reduction=='none':
            return loss
        if self.reduction=='sum':
            return loss.sum()




@MODELS.register
class BCELoss(nn.Module):
    '''二分类交叉熵损失 sigmoid + bceloss
    '''
    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction=reduction
        self.loss = nn.BCEWithLogitsLoss(reduction='none')


    def forward(self, pred, target):
        """
        """
        loss = self.loss(pred, target)
        if self.reduction=='mean':
            return loss.mean()
        if self.reduction=='none':
            return loss
        if self.reduction=='sum':
            return loss.sum()








@MODELS.register
class FocalLoss(nn.Module):
    '''FocalLoss
    '''
    def __init__(self, reduction='mean', gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
        self.reduction=reduction
        self.gamma = gamma
        self.alpha = alpha


    def forward(self, pred, target):
        """
        """
        loss = self.loss(pred, target)
        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction=='mean':
            return loss.mean()
        if self.reduction=='none':
            return loss
        if self.reduction=='sum':
            return loss.sum()







@MODELS.register
class QFocalLoss(nn.Module):
    '''QFocalLoss
    '''
    def __init__(self, reduction='mean', gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
        self.reduction=reduction
        self.gamma = gamma
        self.alpha = alpha


    def forward(self, pred, target):
        """
        """
        loss = self.loss(pred, target)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        modulating_factor = torch.abs(target - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction=='mean':
            return loss.mean()
        if self.reduction=='none':
            return loss
        if self.reduction=='sum':
            return loss.sum()






@MODELS.register
class GIoULoss(nn.Module):
    '''L2损失
    '''
    def __init__(self, reduction='mean'):
        super(GIoULoss, self).__init__()
        self.reduction = reduction


    def forward(self, pred, target):
        """
        """
        giou = self.giou(pred, target)
        loss = 1. - giou
        if self.reduction=='mean':
            return loss.mean()
        if self.reduction=='none':
            return loss
        if self.reduction=='sum':
            return loss.sum()
        

    def giou(self, preds, targets):
        '''计算GIoU(preds, targets均是原始的非归一化距离(ltrb, 注意不是坐标))
            Args:
                preds:   shape=[total_anchor_num, 4(l, t, r, b)]
                targets: shape=[total_anchor_num, 4(l, t, r, b)] 
            Returns:
                giou:  shape=[total_anchor_num, 4]
        '''
        # 左上角和右下角
        lt_min = torch.min(preds[:, :2], targets[:, :2])
        rb_min = torch.min(preds[:, 2:], targets[:, 2:])
        # 重合面积计算
        wh_min = (rb_min + lt_min).clamp(min=0)
        overlap = wh_min[:, 0] * wh_min[:, 1]#[n]
        # 预测框面积和实际框面积计算
        area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
        area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
        # 计算交并比
        union = (area1 + area2 - overlap)
        iou = overlap / (union + 1e-7)
        # 计算外包围框
        lt_max = torch.max(preds[:, :2],targets[:, :2])
        rb_max = torch.max(preds[:, 2:],targets[:, 2:])
        wh_max = (rb_max + lt_max).clamp(0)
        G_area = wh_max[:, 0] * wh_max[:, 1]
        # 计算GIOU
        giou = iou - (G_area - union) / G_area.clamp(1e-10)
        return giou



