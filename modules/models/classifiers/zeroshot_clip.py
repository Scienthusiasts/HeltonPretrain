import torch
import torch.nn as nn
import timm
import torch.distributed as dist
import numpy as np
# 注册机制
from register import MODELS





@MODELS.register
class ZeroShotCLIP(nn.Module):
    """CLIP for Zero-Shot Image Classification
    """
    def __init__(self, clip_model:nn.Module, cat_names, template_prompt):
        """
        Args:
            device:  
            clip_model:      CLIP 视觉基础模型  
            cat_names:       类别名称列表 list[catname1, ..., catnamen]
            template_prompt: 将类别转化为依据描述的prompt模板(待替换部分以{}占位) [str1, ..., strn]
        """
        super().__init__()
        # 基础CLIP模型
        self.clip_model = clip_model.eval()
        self.cat_names = cat_names
        self.nc = len(self.cat_names)
        self.template_prompt = template_prompt
        self.template_num = len(self.template_prompt)
        self.cat_prompts = self._gen_cat_prompts()
        self.cat_embs = None


    def forward(self, x):
        """前向 
        Args:  
            x: 图像 [B, 3, H, W]
        Returns:
            pred:  模型预测结果(logits, 未经过softmax)
        """
        if self.cat_embs == None:
            self.cat_embs = self.clip_model(x.device, self.cat_prompts, modality='text')
            self.cat_embs = self.cat_embs.reshape(self.nc, self.template_num, -1)
        img_embs = self.clip_model(x.device, x)
        # 图文计算余弦相似度, 相似度作为分类logits
        pred_logits = self._cosine_similarity(img_embs, self.cat_embs)

        return pred_logits


    def _gen_cat_prompts(self,):
        '''给定类别和prompt模板, 生成类别prompt
        '''
        # 对于每个类别名, 根据给定的模板生成该类别的prompt
        cat_prompts = []
        for cat_name in self.cat_names:
            cat_prompts += [template.replace("{}", cat_name) for template in self.template_prompt]
        return cat_prompts


    # 计算图像和文本的余弦相似度
    def _cosine_similarity(self, img_f, text_f, scale=100.):
        """计算图像和文本的余弦相似度
        Args: 
            img_f:  [B, D]
            text_f: [nc, N, D]
        Returns:
            logits: [B, nc]

        """
        # 特征向量归一化
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        # 计算余弦相似度
        logits = scale * torch.einsum('bd,cnd->bnc', img_f, text_f)
        # 所有prompt的相似度ensemble [B, N, nc] -> [B, nc]
        logits = torch.mean(logits, dim=1)
        return logits
    



