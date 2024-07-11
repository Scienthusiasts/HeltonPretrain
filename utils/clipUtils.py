import torch
import clip
import os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import shutil
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






def CLIPZeroShot(device, img_dir, prompt, filter_dir):
    '''利用CLIP zeroshot能力来筛选图片'''
    model, preprocess = clip.load("F:/DeskTop/git/HeltonPretrain/ckpt/CLIP_ViT-L-14.pt", device=device)

    for img_name in tqdm(os.listdir(img_dir)):
        if img_name.find('FlickBreeds') == -1:continue
        img_path = os.path.join(img_dir, img_name)
        image = Image.open(img_path)
        image = preprocess(image).unsqueeze(0).to(device)
        text = clip.tokenize(prompt).to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            pred_label = torch.argmax(logits_per_image, dim=1)[0].item()
            # 过滤不满足条件的图像
            if pred_label:
                # print(logits_per_image[0].item(), img_path)
                shutil.move(img_path, os.path.join(filter_dir, img_name))


            
if __name__ == '__main__':
    img_dir = 'E:/datasets/Classification/HUAWEI_cats_dogs_fine_grained/cats_vs_dogs_merge/cat_data_merge'
    filter_dir = 'E:/datasets/Classification/HUAWEI_cats_dogs_fine_grained/filter_cat_data_merge'
    prompt = ["a picture of cat in the scene.", "no cat in the scene."]

    CLIPZeroShot('cuda', img_dir, prompt, filter_dir)