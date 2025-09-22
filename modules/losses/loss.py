import torch
import torch.nn as nn
from register import MODELS



@MODELS.register
class CELoss(nn.Module):
    '''CELoss 交叉熵损失
    '''
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, pred, label):
        """
        Args:
            pred:  [B, cls_num]
            label: [B, ]
            
        """
        loss = self.loss(pred, label)
        return loss
