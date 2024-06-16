import torch
import clip
import os
import torch.nn as nn
import numpy as np
from PIL import Image





# 计算图像和文本的余弦相似度
def COSSim(img_f, text_f, scale=100.):
    # 特征向量归一化
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    text_f = text_f / text_f.norm(dim=-1, keepdim=True)
    # 计算余弦相似度
    logits = scale * img_f @ text_f.t()
    # return logits.softmax(dim=-1)
    return logits










def genLabel(prompt:list[str], cat_ids:list[int], cat_names:list[str]):
    '''给定类别和prompt模板, 生成类别prompt
    '''
    rand_id = np.random.randint(0, high=len(prompt), size=len(cat_ids))
    prompt_labels = [prompt[rand_id[i]].replace("/=/", cat_names[cat_ids[i]]) for i in range(len(cat_ids))]
    return prompt_labels