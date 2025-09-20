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
    def __init__(self, layers_dim:list):
        '''网络初始化
            Args:
                layers_dim:输入输出的维度大小(输入维度必须是[B, C])
            Returns:
                None
        '''
        super(MLPHead, self).__init__()
        '''损失函数'''
        self.clsLoss = nn.CrossEntropyLoss()
        '''网络组件'''
        layers = []
        for i in range(len(layers_dim) - 1):
            in_dim = layers_dim[i]
            out_dim = layers_dim[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            # 最后一层只保留 Linear
            if i < len(layers_dim) - 2:  
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)
        # 权重初始化
        init_weights(self.mlp, 'normal', 0.01)


    def forward(self, x:torch.tensor):
        '''前向传播
        '''
        return self.mlp(x)
    








# for test only:
if __name__ == '__main__':
    x = torch.randn(2, 2048)
    mlp = MLPHead([2048, 256, 10])
    print(mlp)
    out = mlp(x)
    print(out.shape)  # torch.Size([2, 10])