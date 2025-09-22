import os                                   
import numpy as np
from tabulate import tabulate
import torch
import torch.nn as nn
import cv2
import json
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import random
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import rcParams


def showComMatrix(trueList, predList, cat, evalDir):
    '''可视化混淆矩阵
    Args:
        trueList:  验证集的真实标签
        predList:  网络预测的标签
        cat:       所有类别的字典
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








def clacPRCurve(trueList, softList, clsNum):
    '''所有类别下的PR曲线值（基于 sklearn）
    Args:
        - trueList:  验证集的真实标签 (shape: [N])
        - softList:  网络输出的 softmax 概率 (shape: [N, C])
        - clsNum:    类别数

    Returns:
        - PRs:       [clsNum] 列表, 每个元素是 (precision, recall, thresholds)
    '''
    PRs = []
    print('calculating PR per classes...')
    for cls in trange(clsNum):
        label = (trueList == cls).astype(int)
        scores = softList[:, cls]
        precision, recall, thresholds = precision_recall_curve(label, scores)
        PRs.append((precision, recall, thresholds))
    return PRs



def drawPRCurve(cat, trueList, softList, evalDir):
    '''
    绘制类别的PR曲线 
    Args:
        - cat:       类别索引列表
        - trueList:  验证集的真实标签
        - softList:  网络预测的置信度
        - evalDir:   PR曲线图保存路径
    '''
    plt.figure(figsize=(12, 9))
    # 计算所有类别下的PR曲线值
    PRs = clacPRCurve(trueList, softList, len(cat))

    for i, (precision, recall, _) in enumerate(PRs):
        name = cat[i] if cat is not None else f"Class {i}"
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    # 保存图像 
    plt.savefig(os.path.join(evalDir, '类别PR曲线.png'), dpi=200)
    plt.clf()  
    return PRs





def clacAP(PRs, cat, trueList, softList):
    """
    计算每个类别的 AP 和最大 F1Score (基于 sklearn 的 average_precision_score)
        Args:
            - PRs:       clacPRCurve 返回的结果 [(precision, recall, thresholds), ...]
            - cat:       类别名称列表
            - trueList:  验证集真实标签 (shape: [N]) 
            - softList:  网络输出的 softmax 概率 (shape: [N, C])

        Returns:
            - mAP: 平均 AP
            - mF1: 平均最大 F1
            - table_str: tabulate 格式表格
    """

    form = [['Category', 'AP', 'Max_F1Score']]
    mAP = 0.0
    mF1 = 0.0

    for i, cname in enumerate(cat):
        precision, recall, thresholds = PRs[i]

        # --- AP 用 sklearn 官方 average_precision_score ---
        label = (trueList == i).astype(int)
        scores = softList[:, i]
        AP = average_precision_score(label, scores)

        # --- F1: 从 PR 曲线点计算最大值 ---
        eps = 1e-12
        F1_scores = 2 * precision * recall / (precision + recall + eps)
        F1_scores = np.nan_to_num(F1_scores, nan=0.0)
        max_F1 = float(np.max(F1_scores))

        form.append([cname, float(AP), max_F1])
        mAP += AP
        mF1 += max_F1

    mAP /= len(cat)
    mF1 /= len(cat)
    form.append(['average', mAP, mF1])

    return mAP, mF1, tabulate(form, headers='firstrow')