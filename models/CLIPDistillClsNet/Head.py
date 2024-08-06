import torch
import torch.nn as nn
import timm
from torchvision import models

from models.YOLOBlock import *
from utils.utils import *
from models import CLIP
from loss.loss import InfoNCELoss, TripleLoss, ThresholdMarginLoss, IdClassifyLoss



class Head(nn.Module):
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
        super(Head, self).__init__()
        self.cat_nums = cls_num
        self.clip_embedding_c = clip_embedding_c
        '''损失函数'''
        self.clsLoss = nn.CrossEntropyLoss()
        self.contrastLoss = InfoNCELoss()
        self.tripleLoss = TripleLoss(margin=1, p=2)
        self.distillLoss = nn.SmoothL1Loss()
        # 针对可学习动态阈值设定的损失
        self.TMarginLoss = ThresholdMarginLoss()
        self.idClassifyLoss = IdClassifyLoss()

        '''网络组件'''
        # CLIPModel定义为全局变量, 而不是类成员
        global CLIPModel
        CLIPModel = clip_model
        # 特征提取
        if add_share_head:
            self.share_head = nn.Sequential(
                Conv(input_c , mid_c[0], kernel_s[0], 1, 0),
                Conv(mid_c[0], mid_c[1], kernel_s[1], 1, 0),
                Conv(mid_c[1], mid_c[2], kernel_s[2], 1, 0),
            )
            # 分类头
            self.cls_head = nn.Linear(mid_c[2], self.cat_nums)
            self.clip_head = nn.Linear(mid_c[2], clip_embedding_c)
        else:
            self.share_head = nn.Identity()
            self.cls_head = nn.Linear(input_c, self.cat_nums)
            self.clip_head = nn.Linear(input_c, clip_embedding_c)

        # 无论最后尺寸多大，都池化成1x1,这样输入的图像尺寸就可以是任意大小,但必须大于224x224
        self.gap = nn.AdaptiveAvgPool2d(1)
        '''可学习动态阈值'''
        # self.lernable_T = nn.Sequential(
        #     nn.BatchNorm1d(clip_embedding_c*2),
        #     nn.Linear(clip_embedding_c*2, 2),
        # )
        # 权重初始化
        init_weights(self.share_head, 'normal', 0, 0.01)
        init_weights(self.cls_head, 'normal', 0, 0.01)
        init_weights(self.clip_head, 'normal', 0, 0.01)
        # init_weights(self.lernable_T, 'normal', 0, 0.01)



    def forward(self, x:torch.tensor):
        '''前向传播
        '''
        x = self.share_head(x)
        x = self.gap(x)
        x = x.reshape(x.shape[0], -1)
        cls_logits = self.cls_head(x)
        embeddings = self.clip_head(x)
        return cls_logits, embeddings


    def forward_shared(self, x:torch.tensor):
        '''共享层前向传播
        '''
        x = self.share_head(x)
        x = self.gap(x)
        x = x.reshape(x.shape[0], -1)
        return x


    def batchLoss(self, combined_x, combined_clip_imgs, batch_labels):
        # combined_x = [batch_imgs, anchor_imgs, pos_imgs, neg_imgs]
        # combined_clip_imgs = [batch_clip_imgs, anchor_imgs, pos_imgs, neg_imgs]
        bs = combined_clip_imgs.shape[0] // 4
        # 先经过共享头进一步提取特征
        combined_x_feature = self.forward_shared(combined_x)
        # 经过编码头提取编码特征
        combined_embeddings = self.clip_head(combined_x_feature)
        # 经过分类头提取分类特征
        cls_logits = self.cls_head(combined_x_feature[:bs])
        img_embeddings, anchor_embeddings, pos_embeddings, neg_embeddings = torch.split(combined_embeddings, bs, dim=0)
        # clip图像编码
        combined_clip_embeddings = CLIPModel.forwardImg(combined_clip_imgs)

        '''常规分类损失'''
        cls_loss = self.clsLoss(cls_logits, batch_labels)
        '''对比损失'''
        contrast_loss = self.tripleLoss(anchor_embeddings, pos_embeddings, neg_embeddings)
        '''蒸馏损失'''
        distill_loss = self.distillLoss(combined_embeddings, combined_clip_embeddings)
        '''图文匹配对比损失'''
        img_text_contrast_loss = self.contrastLoss(img_embeddings, CLIPModel.prompts_embeddings_train.float().detach(), batch_labels)
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











    def batchLoss_(self, combined_x, combined_clip_imgs, batch_labels):
        bs = combined_clip_imgs.shape[0] // 4
        # 前向
        combined_x_cls_logits, combined_x_embeddings = self.forward(combined_x)
        '''常规分类损失'''
        cls_loss = self.clsLoss(combined_x_cls_logits, batch_combined_labels)
        '''对比损失'''
        # 拆分特征,之前是图像和对比增强图像拼在一起前向, 现在拆分开
        x_embeddings, contrast_x_embeddings = torch.split(combined_x_embeddings, bs, dim=0)
        contrast_label = torch.arange(0, bs).to(combined_x.device)
        contrast_loss = self.contrastLoss(x_embeddings, contrast_x_embeddings, contrast_label)
        '''Triplet Loss (对比损失)'''
        # triplet_loss = self.tripletLoss(x_embeddings, contrast_x_embeddings)
        '''蒸馏损失'''
        prompts_token_train = CLIPModel.genTrainLabel()
        img_embeddings, text_embeddings = CLIPModel.forward(batch_clip_imgs, prompts_token_train)
        img_embeddings = img_embeddings.float()
        text_embeddings = text_embeddings.float()
        distill_loss = self.distillLoss(x_embeddings, img_embeddings)
        '''图文匹配对比损失'''
        batch_labels = batch_combined_labels[:bs]
        img_text_contrast_loss = self.contrastLoss(x_embeddings, text_embeddings, batch_labels)
        '''可学习动态阈值的损失'''
        # # [bs, 768*2]
        # # 首先扩展成二维concat矩阵(样本与增强样本之间两两concat)
        # embeddings_expanded = x_embeddings.unsqueeze(1)
        # embeddings_expanded = embeddings_expanded.expand(bs, bs, self.clip_embedding_c)
        # contrastx_embeddings_expanded = contrast_x_embeddings.unsqueeze(0)
        # contrastx_embeddings_expanded = contrastx_embeddings_expanded.expand(bs, bs, self.clip_embedding_c)
        # cat_M = torch.cat((embeddings_expanded, contrastx_embeddings_expanded), dim=-1)
        # # 计算两两的动态阈值
        # cat_M = cat_M.reshape(-1, self.clip_embedding_c*2)
        # learnable_T = self.lernable_T(cat_M).reshape(bs, bs)
        # # 计算损失
        # T_margin_loss = self.TMarginLoss(learnable_T, x_embeddings, contrast_x_embeddings)
        '''个体识别分类损失'''
        # # [bs, 768*2]
        # # 首先扩展成二维concat矩阵(样本与增强样本之间两两concat)
        # embeddings_expanded = x_embeddings.unsqueeze(1)
        # embeddings_expanded = embeddings_expanded.expand(bs, bs, self.clip_embedding_c)
        # contrastx_embeddings_expanded = contrast_x_embeddings.unsqueeze(0)
        # contrastx_embeddings_expanded = contrastx_embeddings_expanded.expand(bs, bs, self.clip_embedding_c)
        # cat_M = torch.cat((embeddings_expanded, contrastx_embeddings_expanded), dim=-1)
        # # 计算两两的个体识别分数logits
        # cat_M = cat_M.reshape(-1, self.clip_embedding_c*2)
        # learnable_T = self.lernable_T(cat_M)
        # # 构造标签
        # target_GT = torch.eye(bs).type(torch.LongTensor).to(learnable_T.device).reshape(-1)
        # # 计算损失
        # id_classify_loss = self.idClassifyLoss(learnable_T, target_GT)
        '''总损失'''
        # total_loss = cls_loss + contrast_loss + distill_loss * 100 + img_text_contrast_loss + id_classify_loss + T_margin_loss + triplet_loss
        total_loss = cls_loss + contrast_loss + distill_loss * 100 + img_text_contrast_loss # + id_classify_loss + T_margin_loss + triplet_loss
        # total_loss = T_margin_loss
        '''损失以字典形式组织'''
        loss = dict(
            total_loss = total_loss,
            cls_loss = cls_loss,
            contrast_loss = contrast_loss,
            distill_loss = distill_loss,
            img_text_contrast_loss = img_text_contrast_loss,
            # soft_triplet_loss = triplet_loss,
            # T_margin_loss = T_margin_loss,
            # id_classify_loss = id_classify_loss,
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