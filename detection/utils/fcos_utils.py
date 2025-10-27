import numpy as np
import torch
from torch.nn import functional as F
import cv2
import torch.nn as nn
import os
from tqdm import tqdm
import matplotlib.pyplot as plt






def reshape_cat_out(inputs):
    '''将不同尺度的预测结果拼在一起
    '''
    out=[]
    for pred in inputs:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [inputs[0].shape[0], -1, inputs[0].shape[1]])
        out.append(pred)
    return torch.cat(out, dim=1)








def vis_FCOS_heatmap(cls_logits, cnt_logits, ori_shape, input_shape, image, save_vis_path=None):
    '''可視化 FCOS obj_heatmap
        Args:
            - predicts:    多尺度特征圖
            - ori_shape:   原圖像尺寸
            - input_shape: 网络接收的尺寸
            - padding:     输入网络时是否灰边填充处理
        Returns:
    '''
    W, H = ori_shape
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([[0.229, 0.224, 0.225]]) 
    image = image.squeeze(0).permute(1,2,0).cpu().numpy()
    image = ((image * std + mean)*255).astype('uint8')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 对三个尺度特征图分别提取 obj heatmap
    for layer in range(len(cnt_logits)):
        cnt_logit = cnt_logits[layer].cpu()
        cls_logit = cls_logits[layer].cpu()
        b, c, h, w = cnt_logit.shape
        # [bs=1,1,w,h] -> [w,h]
        cnt_logit = cnt_logit[0,0,...]
        # [bs=1,w,h] -> [w,h]
        cls_logit = cls_logit[0,...]
        '''提取centerness-map(类别无关的obj置信度)'''
        saveVisCenternessMap(cnt_logit, image, W, H, layer, input_shape, save_vis_path)
        '''提取类别最大置信度heatmap(类别最大置信度*centerness)'''
        # saveVisScoreMap(cnt_logit, cls_logit, image, W, H, layer, input_shape, save_vis_path)



def saveVisCenternessMap(cnt_logit, image, W, H, layer, input_shape, save_vis_path):
    '''提取objmap(类别无关的obj置信度)
    '''
    # 取objmap, 并执行sigmoid将value归一化到(0,1)之间
    heat_map = F.sigmoid(cnt_logit).numpy()
    # resize到网络接受的输入尺寸
    heat_map = cv2.resize(heat_map, (input_shape[0], input_shape[1]))
    heatmap2Img(heat_map, image, W, H, layer, input_shape, save_vis_path)



def saveVisScoreMap(cnt_logit, cls_logit, image, W, H, layer, input_shape, save_vis_path):
    '''提取类别最大置信度heatmap(类别最大置信度*centerness)
    '''
    # 取objmap, 并执行sigmoid将value归一化到(0,1)之间
    centerness_map = F.sigmoid(cnt_logit).numpy()
    cat_score_map = F.sigmoid(cls_logit).numpy()
    cat_score_map = np.max(cat_score_map, axis=0)
    heat_map = centerness_map * cat_score_map
    heatmap2Img(heat_map, image, W, H, layer, input_shape, save_vis_path)



def heatmap2Img(heat_map, image, W, H, layer, input_shape, save_vis_path):
    '''heatmap -> img -> save'''
    # resize到网络接受的输入尺寸
    heat_map = cv2.resize(heat_map, (input_shape[0], input_shape[1]))
    heat_map = (heat_map * 255).astype('uint8')
    # resize到原图尺寸
    heat_map = cv2.resize(heat_map, (W, H))
    # 灰度转伪彩色图像
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    # heatmap和原图像叠加显示
    heatmap_img = cv2.addWeighted(heat_map, 0.5, image, 0.5, 0)
    # 保存
    if save_vis_path!=None:
        save_dir, save_name = os.path.split(save_vis_path)
        save_name = f'heatmap{layer}_' + save_name
        cv2.imwrite(os.path.join(save_dir, save_name), heatmap_img)



