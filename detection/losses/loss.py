import torch
import torch.nn as nn
import math
from heltonx.utils.register import MODELS





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
class IoULoss(nn.Module):
    '''L2损失
    '''
    def __init__(self, iou_type, xywh=False, reduction='mean'):
        super(IoULoss, self).__init__()
        self.reduction = reduction
        self.iou_type = iou_type
        self.xywh = xywh
        self.eps = 1e-7


    def forward(self, pred, target):
        """
        """
        iou = self.bbox_iou_pairwise(pred, target)
        loss = 1. - iou
        if self.reduction=='mean':
            return loss.mean()
        if self.reduction=='none':
            return loss
        if self.reduction=='sum':
            return loss.sum()
        
    def bbox_iou_pairwise(self, box1, box2):
        """计算 box1 和 box2 的 IoU (对应位置一对一计算)
            Args:
                box1: [total_anchor_num, 4(x, y, w, h / x0, y0, x1, y1)]
                box2: [total_anchor_num, 4(x, y, w, h / x0, y0, x1, y1)]
            Returns:
                iou:  [total_anchor_num]
        """
        if self.xywh:  # (x, y, w, h) → (x1, y1, x2, y2)
            x1, y1, w1, h1 = box1.unbind(-1)
            x2, y2, w2, h2 = box2.unbind(-1)
            b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
            b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
            b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
            b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
        else:  # (x1, y1, x2, y2)
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

        # 相交区域
        inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
        inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        inter = inter_w * inter_h
        # 各自面积
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        # 并集面积
        union = area1 + area2 - inter + self.eps
        # IoU
        iou = inter / union

        # 处理 GIoU / DIoU / CIoU
        if self.iou_type in ["giou", "diou", "ciou"]:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # 包围盒宽度
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # 包围盒高度

            if self.iou_type in ["diou", "ciou"]:
                c2 = cw ** 2 + ch ** 2 + self.eps
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)**2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2)**2) / 4
                if self.iou_type == "ciou":
                    v = (4 / math.pi**2) * (torch.atan((b2_x2 - b2_x1) / (b2_y2 - b2_y1 + self.eps)) -
                                            torch.atan((b1_x2 - b1_x1) / (b1_y2 - b1_y1 + self.eps)))**2
                    with torch.no_grad():
                        alpha = v / (v - iou + 1 + self.eps)
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
                return iou - rho2 / c2  # DIoU
            # GIoU
            c_area = cw * ch + self.eps
            return iou - (c_area - union) / c_area
        # IoU
        return iou