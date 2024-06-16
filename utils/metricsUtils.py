import os                                   
import numpy as np
from tabulate import tabulate
import torch
import torch.nn as nn
import cv2
import json
import importlib
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import rcParams

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)










def showComMatrix(trueList, predList, cat, evalDir):
    '''可视化混淆矩阵

    Args:
        - trueList:  验证集的真实标签
        - predList:  网络预测的标签
        - cat:       所有类别的字典

    Returns:
        None
    '''
    if len(cat)>=50:
        # 100类正合适的大小  
        plt.figure(figsize=(40, 33))
        plt.subplots_adjust(left=0.05, right=1, bottom=0.05, top=0.99) 
    else:
        # 10类正合适的大小
        plt.figure(figsize=(12, 9))
        plt.subplots_adjust(left=0.1, right=1, bottom=0.1, top=0.99) 

    conf_mat = confusion_matrix(trueList, predList)
    df_cm = pd.DataFrame(conf_mat, index=cat, columns=cat)
    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap=plt.cm.Blues)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, ha = 'right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation = 50, ha = 'right')
    plt.ylabel('true label')
    plt.xlabel('pred label')
    if not os.path.isdir(evalDir):os.makedirs(evalDir)
    # 保存图像
    plt.savefig(os.path.join(evalDir, '混淆矩阵.png'), dpi=200)
    plt.clf() 








def calcPRThreshold(trueList, softList, clsNum, T):
    '''给定一个类别, 单个阈值下的PR值

    Args:
        - trueList:  验证集的真实标签
        - predList:  网络预测的标签
        - clsNum:    类别索引

    Returns:
        precision, recall
    '''
    label = (trueList==clsNum)
    prob = softList[:,clsNum]>T
    TP = sum(label*prob)   # 正样本预测为正样本
    FN = sum(label*~prob)  # 正样本预测为负样本
    FP = sum(~label*prob)  # 负样本预测为正样本
    precision = TP / (TP + FP) if (TP + FP)!=0 else 1
    recall = TP / (TP + FN) 
    return precision, recall, T







def clacPRCurve(trueList, softList, clsNum, interval=100):
    '''所有类别下的PR曲线值

    Args:
        - trueList:  验证集的真实标签
        - predList:  网络预测的标签
        - clsNum:    类别索引列表
        - interval:  阈值变化划分的区间，如interval=100, 则间隔=0.01

    Returns:
        - PRs:       不同阈值下的PR值[2, interval, cat_num]
    '''
    PRs = []
    print('calculating PR per classes...')
    for cls in trange(clsNum):
        PR_value = [calcPRThreshold(trueList, softList, cls, i/interval) for i in range(interval+1)]
        PRs.append(np.array(PR_value))

    return np.array(PRs)








def drawPRCurve(cat, trueList, softList, evalDir):
    '''绘制类别的PR曲线 

    Args:
        - cat:  类别索引列表
        - trueList:  验证集的真实标签
        - softList:  网络预测的置信度
        - evalDir:   PR曲线图保存路径

    Returns:
        None
    '''
    
    plt.figure(figsize=(12, 9))
    # 计算所有类别下的PR曲线值
    PRs = clacPRCurve(trueList, softList, len(cat))
    # 绘制每个类别的PR曲线
    for i in range(len(cat)):
        PR = PRs[i]
        plt.plot(PR[:,1], PR[:,0], linewidth=1)
    plt.legend(labels=cat)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim(0,1)
    plt.ylim(0,1)
    # 保存图像 
    plt.savefig(os.path.join(evalDir, '类别PR曲线.png'), dpi=200)
    plt.clf()  
    return PRs








def clacAP(PRs, cat):
    '''计算每个类别的 AP, F1Score

    Args:
        - PRs:  不同阈值下的PR值[2, interval, cat_num]
        - cat:  类别索引列表

    Returns:
        None
    '''
    form = [['catagory', 'AP', 'F1_Score']]
    # 所有类别的平均AP与平均F1Score
    mAP, mF1Score = 0, 0
    for i in range(len(cat)):
        PR = PRs[i]
        AP = 0
        for j in range(PR.shape[0]-1):
            # 每小条梯形的矩形部分+三角形部分面积
            h = PR[j, 0] - PR[j+1, 0]
            w = PR[j, 1] - PR[j+1, 1]
            AP += (PR[j+1, 0] * w) + (w * h / 2)

            if(PR[j, 2]==0.5):
                F1Score0_5 = 2 * PR[j, 0] * PR[j, 1] / (PR[j, 0] + PR[j, 1] + 1e-7)

        form.append([cat[i], AP, F1Score0_5])  
        mAP += AP
        mF1Score += F1Score0_5

    mAP /= len(cat)
    mF1Score /= len(cat)
    form.append(['average',mAP, mF1Score]) 

    return mAP, mF1Score, tabulate(form, headers='firstrow') # tablefmt='fancy_grid'


