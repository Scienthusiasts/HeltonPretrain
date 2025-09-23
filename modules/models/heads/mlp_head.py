import torch
import torch.nn as nn
from torchvision import models
from utils.utils import init_weights
# 注册机制
from register import MODELS


@MODELS.register
class MLPHead(nn.Module):
    '''Head
    '''
    def __init__(self, layers_dim:list, cls_loss:nn.Module):
        '''网络初始化
            Args:
                layers_dim: 输入输出的维度大小, 例[C1, C2, ..., Cn]有n-1层Linear, 第i层维度是Ci+1
                cls_loss:   分类损失实例
            Returns:
                None
        '''
        super(MLPHead, self).__init__()
        '''网络组件'''
        self.mlp = self.make_layers(layers_dim)
        '''损失函数'''
        self.clsLoss = cls_loss
        # 权重初始化
        init_weights(self.mlp, 'normal', 0.01)


    def make_layers(self, layers_dim):
        '''根据layers_dim生成自定义MLP层
        '''
        layers = []
        for i in range(len(layers_dim) - 1):
            in_dim = layers_dim[i]
            out_dim = layers_dim[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            # 最后一层只保留 Linear
            if i < len(layers_dim) - 2:  
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)


    def forward(self, x):
        '''前向传播
            x: 输入维度必须是[B, C]
        '''
        return self.mlp(x)
    

    def loss(self, x, y):
        '''前向传播+计算损失(训练时使用)
            x: 输入维度必须是[B, C]
        '''
        pred = self.forward(x)
        cls_loss = self.clsLoss(pred, y)

        # 顺便计算并返回acc.指标
        pred_scores, pred_labels = torch.max(pred, dim=1)
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
    x = torch.randn(2, 2048)
    mlp = MLPHead([2048, 256, 10])
    print(mlp)
    out = mlp(x)
    print(out.shape)  # torch.Size([2, 10])