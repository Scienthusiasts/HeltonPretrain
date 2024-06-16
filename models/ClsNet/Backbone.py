import torch
import torch.nn as nn
import timm
from torchvision import models
import os




'''
huggingface里的timm模型:
https://huggingface.co/timm?sort_models=downloads#models
'''


class Backbone(nn.Module):
    '''Backbone
    '''
    def __init__(self, backbone_name:str, loadckpt=False, pretrain=True, froze=True):
        '''网络初始化

        Args:
            :param modelType: 使用哪个模型(timm库里的模型)
            :param loadckpt:  是否导入模型权重(是则输入权重路径)
            :param pretrain:  是否用预训练模型进行初始化
            :param froze:     是否冻结Backbone

        Returns:
            None
        '''
        super(Backbone, self).__init__()
        # 模型接到线性层的维度

        # 加载模型(features_only=True 只加载backbone部分)
        # features_only=True 只提取特征图(5层)，不加载分类头
        self.backbone = timm.create_model(backbone_name, pretrained=pretrain, features_only=True)
        # 是否冻结backbone
        if froze:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        # 是否导入预训练权重
        if loadckpt: 
            self.load_state_dict(torch.load(loadckpt))
            print(f'backbone pretrain ckpt({backbone_name}) loaded! ')

    def forward(self, x):
        '''前向传播
        '''
        x = self.backbone(x)
        return x[-1]
















# for test only
if __name__ == '__main__':
    # mobilenetv3_large_100.ra_in1k  resnet50.a1_in1k  darknetaa53.c2ns_in1k cspdarknet53.ra_in1k cspresnext50.ra_in1k
    backbone_name = 'mobilenetv3_large_100.ra_in1k'
    backbone = Backbone(backbone_name=backbone_name, pretrain=True)
    torch.save(backbone.state_dict(), f"ckpt/{backbone_name}_w_sharehead.pt")
    # 验证 2
    # print(backbone)
    x = torch.rand((4, 3, 224, 224))
    out = backbone(x)
    print(out.shape)