import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import init_weights
# 注册机制
from register import MODELS


@MODELS.register
class ProtoHead(nn.Module):
    '''Head
    '''
    def __init__(self, layers_dim:list, nc:int, cls_loss:nn.Module):
        '''网络初始化
            Args:
                layers_dim: 输入输出的维度大小, 例[C1, C2, ..., Cn]有n-1层1x1conv, 第i层维度是Ci+1
                nc:         分类类别数
                cls_loss:   分类损失实例
            Returns:
                None
        '''
        super(ProtoHead, self).__init__()
        '''网络组件'''
        self.mlp = self.make_layers(layers_dim)
        self.prototypes = nn.Parameter(torch.randn(nc, layers_dim[-1]))
        '''损失函数'''
        self.clsLoss = cls_loss
        # 权重初始化
        init_weights(self.mlp, 'normal', 0.01)
        # init_weights(self.prototypes, 'normal', 0.01)


    def make_layers(self, layers_dim):
        '''根据layers_dim生成自定义MLP层
        '''
        layers = []
        for i in range(len(layers_dim) - 1):
            in_dim = layers_dim[i]
            out_dim = layers_dim[i + 1]
            layers.append(nn.Conv2d(in_dim, out_dim, 1, bias=False))
            # 最后一层只保留 1x1conv
            if i < len(layers_dim) - 2:  
                layers.append(nn.BatchNorm2d(out_dim))
                layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)


    def forward(self, x):
        '''前向传播
            x: 输入维度必须是[B, C, H, W]
        '''
        # [B, C, H, W] -> [B, dim, H, W] -> [B, dim, H*W]
        x = self.mlp(x).flatten(2) 
        # 计算余弦相似度
        x_norm = F.normalize(x, p=2, dim=1) 
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1) 
        # 使用einsum计算余弦相似度: [B, dim, C], [nc, dim] -> [B, nc, C] -> [B, nc]
        sim = torch.einsum('bdc,nd->bnc', x_norm, prototypes_norm).sum(dim=-1)
        return sim




    def loss(self, x, y):
        '''前向传播+计算损失(训练时使用)
            x: 输入维度必须是[B, C, H, W]
        '''
        logits = self.forward(x)
        cls_loss = self.clsLoss(logits, y)

        # 顺便计算并返回acc.指标
        pred_logits, pred_labels = torch.max(logits, dim=1)
        acc = sum(pred_labels==y) / y.shape[0]

        # 组织成字典形式返回
        losses = dict(
            total_loss = cls_loss,
            cls_loss = cls_loss,
            acc = acc
        )
        return losses









# for test only:
if __name__ == '__main__':
    cls_loss = nn.BCELoss()
    x = torch.randn(4, 2048, 7, 7)
    mlp = ProtoHead([2048, 256, 10], 10, cls_loss)
    print(mlp)
    out = mlp(x)
    print(out.shape)  # torch.Size([4, 10])
    print(out)