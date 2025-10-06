import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import init_weights
# 注册机制
from utils.register import MODELS


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
        nn.init.trunc_normal_(self.prototypes, std=0.02)



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
            Args:
                x: 输入维度必须是[B, C, H, W]
            Returns:
                sim: 逐类别余弦相似度[B, nc]
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
            Args:
                x: 输入维度必须是[B, C, H, W]
                y: 标签[B, ]
            Returns:
                losses: 字典形式组织的损失
        '''
        logits = self.forward(x)
        cls_loss = self.clsLoss(logits, y)
        # 顺便计算并返回acc.指标
        pred_logits, pred_labels = torch.max(logits, dim=1)
        acc = sum(pred_labels==y) / y.shape[0]
        # 组织成字典形式返回
        losses = dict(
            cls_loss = cls_loss,
            acc = acc
        )
        return losses




    def forward_with_protoheatmap(self, x):
        """前向 + 可视化prototype的注意力热图
            Args:
                x: 输入维度必须是[B, C, H, W]
            Returns:
                sim:         逐类别特征图的余弦相似度之和[B, nc]
                sim_heatmap: 归一化余弦相似度特征图, 可以用来可视化
        """
        # 记录形状
        B, _, H, W = x.shape
        nc = self.prototypes.shape[0]
        # [B, C, H, W] -> [B, dim, H, W] -> [B, dim, H*W]
        x = self.mlp(x).flatten(2) 
        # 计算余弦相似度
        x_norm = F.normalize(x, p=2, dim=1) 
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1) 
        # 使用einsum计算余弦相似度: [B, dim, C], [nc, dim] -> [B, nc, C]
        sim_feat = torch.einsum('bdc,nd->bnc', x_norm, prototypes_norm)
        # [B, nc, C] -> [B, nc]
        sim = sim_feat.sum(dim=-1)

        '''归一化'''
        # 范围归一化 [-1, 1] -> [0, 1]
        # sim_heatmap = (sim_feat + 1.) * 0.5
        # 最大最小归一化
        sim_max, sim_min = torch.max(sim_feat, dim=-1, keepdim=True)[0], torch.min(sim_feat, dim=-1, keepdim=True)[0]
        sim_heatmap = (sim_feat - sim_min) / (sim_max - sim_min)
        # [B, nc, C] -> [B, nc, H, W]
        sim_heatmap = sim_heatmap.reshape(B, nc, H, W)
        return sim, sim_heatmap






# for test only:
if __name__ == '__main__':
    cls_loss = nn.BCELoss()
    x = torch.randn(4, 2048, 7, 7)
    mlp = ProtoHead([2048, 256, 10], 10, cls_loss)
    print(mlp)
    out = mlp(x)
    print(out.shape)  # torch.Size([4, 10])
    print(out)