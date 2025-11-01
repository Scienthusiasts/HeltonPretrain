import torch
import torch.nn as nn
from heltonx.utils.register import MODELS



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
            logits: [B, cls_num] / [B, vocab_size, seq_lens]
            labels: [B, ] / [B, seq_lens]
        """
        loss = self.loss(logits, labels)
        return loss
    