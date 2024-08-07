from models.DistillClsNet import DistillClsNet
import torch

import os                                   
import numpy as np
import torch
import torch.nn as nn
import importlib
from tqdm import tqdm, trange
import random
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch.nn.functional as F

from utils.clipUtils import COSSim
from utils.metricsUtils import *
from utils.utils import *
from datasets.preprocess import Transforms







def imgFineInfer(model, device, img_path):
    '''细粒度猫狗图像分类
    '''
    tf = Transforms(imgSize=[224, 224])
    pred_name = inferenceSingleImg(model, device, tf, img_path, return_name=True)
    print('细粒度品种识别结果为: ', pred_name,)


def imgIDInfer(model, device, img_pair_paths):
    '''猫狗个体识别
    '''
    tf = Transforms(imgSize=[224, 224])
    is_pair = identifyPair(model, device, tf, img_pair_paths)
    print('两张图片是否同一个体: ', is_pair)



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = 'id'
    ckpt_path = '_last_half_lp.pt'
    cat_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 
                'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 
                'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 
                'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 
                'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']
    cls_num = len(cat_names)
    backbone_name = 'resnetaa50d.sw_in12k_ft_in1k'
    backbone = dict(loadckpt=False, pretrain=False, froze=True)
    head = dict(add_share_head=True, kernel_s=[1, 1, 1], mid_c=[512, 512, 768], clip_embedding_c=768)
    # 初始化模型
    model = DistillClsNet.Model(cls_num, cat_names, backbone_name, ckpt_path, backbone, head, 'ensemble').to(device)
    model.eval()

    if mode=='fine':
        '''细粒度分类'''
        img_path = '1.jpg'
        imgFineInfer(model, device, img_path)
    elif mode=='id':
        '''个体识别'''
        img_pair_paths = ["1.jpg", "3.jpg"]
        imgIDInfer(model, device, img_pair_paths)
    else:
        print('mode no valid !')



