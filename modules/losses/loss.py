import torch
import torch.nn as nn
from register import MODELS



@MODELS.register
class CELoss(nn.Module):
    '''CELoss 多分类交叉熵损失(适合输出的概率总和=1, 互斥的多分类任务)
    '''
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    
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
    def __init__(self):
        super(MultiClassBCELoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
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