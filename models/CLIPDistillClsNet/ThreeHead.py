import torch
import torch.nn as nn
import timm
from torchvision import models

from models.YOLOBlock import *
from utils.utils import *
from models import CLIP
from loss.loss import InfoNCELoss






class ShareHead(nn.Module):
    def __init__(self, input_c:int, kernel_s:list[int], mid_c:list[int], share_head=True):
        super(ShareHead, self).__init__()
        if share_head:
            self.ConvBlock = nn.Sequential(
                    Conv(input_c , mid_c[0], kernel_s[0], 1, 0),
                    Conv(mid_c[0], mid_c[1], kernel_s[1], 1, 0),
                    Conv(mid_c[1], mid_c[2], kernel_s[2], 1, 0),
                ) 
        else:
            self.ConvBlock = nn.Identity()
        # 无论最后尺寸多大，都池化成1x1,这样输入的图像尺寸就可以是任意大小,但必须大于224x224
        self.gap = nn.AdaptiveAvgPool2d(1)


    def forward(self, x:torch.tensor):
        x = self.ConvBlock(x)
        x = self.gap(x)
        x = x.reshape(x.shape[0], -1)
        return x







class DecoupledHead(nn.Module):
    '''Backbone
    '''
    def __init__(self, cls_num:int, input_c:int, clip_embedding_c:int, kernel_s:list[int], mid_c:list[int], clip_model, add_share_head=True):
        '''网络初始化

        Args:
            - backbone_name:  数据集类别数
            - cls_num:        使用哪个模型(timm库里的模型)
            - input_c:        使用哪个模型(timm库里的模型)
            - kernel_s:       sharehead里每个卷积核大小
            - mid_c:          sharehead里每个卷积通道数大小
            - add_share_head: 是否使用sharehead进一步提取特征
 
        Returns:
            None
        '''
        super(DecoupledHead, self).__init__()
        self.cat_nums = cls_num
        '''损失函数'''
        self.clsLoss = nn.CrossEntropyLoss()
        self.contrastLoss = InfoNCELoss()
        self.distillLoss = nn.SmoothL1Loss()

        '''网络组件'''
        # CLIPModel定义为全局变量, 而不是类成员
        global CLIPModel
        CLIPModel = clip_model
        # 特征提取
        # 有三个任务头:
        self.share_head = nn.Sequential(
                    Conv(input_c , mid_c[0], kernel_s[0], 1, 0),
                    Conv(mid_c[0], mid_c[1], kernel_s[1], 1, 0),
                    Conv(mid_c[1], mid_c[2], kernel_s[2], 1, 0),
                ) 
        self.cls_head = nn.Linear(mid_c[2], self.cat_nums)
        self.clip_head = nn.Linear(mid_c[2], clip_embedding_c)
        self.contrast_head = nn.Linear(mid_c[2], 256)
        # 无论最后尺寸多大，都池化成1x1,这样输入的图像尺寸就可以是任意大小,但必须大于224x224
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 权重初始化
        init_weights(self.share_head, 'normal', 0, 0.01)
        init_weights(self.cls_head, 'normal', 0, 0.01)
        init_weights(self.clip_head, 'normal', 0, 0.01)
        init_weights(self.contrast_head, 'normal', 0, 0.01)



    def forward(self, x:torch.tensor):
        '''前向传播
        '''
        x = self.share_head(x)
        x = self.gap(x)
        x = x.reshape(x.shape[0], -1)
        cls_logits = self.cls_head(x)
        embeddings = self.clip_head(x)
        contrast_embeddings = self.contrast_head(x)
        return cls_logits, embeddings, contrast_embeddings
    



    def batchLoss(self, combined_x, batch_clip_imgs, batch_combined_labels):
        bs = combined_x.shape[0] // 2
        # 前向
        combined_x_cls_logits, combined_x_embeddings, combined_x_contrast_embeddings = self.forward(combined_x)
        '''常规分类损失'''
        cls_loss = self.clsLoss(combined_x_cls_logits, batch_combined_labels)
        '''对比损失'''
        # 拆分特征,之前是图像和对比增强图像拼在一起前向, 现在拆分开
        x_embeddings, contrast_x_embeddings = torch.split(combined_x_contrast_embeddings, bs, dim=0)
        contrast_label = torch.arange(0, bs).to(combined_x.device)
        contrast_loss = self.contrastLoss(x_embeddings, contrast_x_embeddings, contrast_label)
        '''蒸馏损失'''
        x_clip_embeddings, _ = torch.split(combined_x_embeddings, bs, dim=0)
        prompts_token_train = CLIPModel.genTrainLabel()
        img_embeddings, text_embeddings = CLIPModel.forward(batch_clip_imgs, prompts_token_train)
        img_embeddings = img_embeddings.float()
        text_embeddings = text_embeddings.float()
        distill_loss = self.distillLoss(x_clip_embeddings, img_embeddings)
        '''图文匹配对比损失'''
        batch_labels = batch_combined_labels[:bs]
        img_text_contrast_loss = self.contrastLoss(x_clip_embeddings, text_embeddings, batch_labels)
        '''总损失'''
        total_loss = cls_loss + contrast_loss + distill_loss * 100 + img_text_contrast_loss 
        '''损失以字典形式组织'''
        loss = dict(
            total_loss = total_loss,
            cls_loss = cls_loss,
            contrast_loss = contrast_loss,
            distill_loss = distill_loss,
            img_text_contrast_loss = img_text_contrast_loss,
        )
        return loss









# for test only:
if __name__ == '__main__':
    from models.CLIPDistillClsNet.Backbone import Backbone

    cls_num = 20
    backbone_name='cspresnext50.ra_in1k'
    mid_c = [512, 256, 256]
    backbone = Backbone(backbone_name, pretrain=False)
    head = Head(backbone_name, cls_num, mid_c)

    # 验证 
    x = torch.rand((4, 3, 224, 224))
    backbone_feat = backbone(x)
    out = head(backbone_feat)
    print(out.shape)