import numpy as np
import torch
from PIL import Image
import torch.utils.data.dataset as data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import albumentations as A
import os
import cv2

# 自定义
from utils.utils import seed_everything
from datasets.preprocess import Transforms





class ClsDataset(data.Dataset):      
    '''有监督分类任务对应的数据集读取方式
    '''
    def __init__(self, dir, mode, imgSize, contrast=False):    
        '''__init__() 为默认构造函数，传入数据集类别（训练或测试），以及数据集路径

        Args:
            :param dir:     图像数据集的根目录
            :param mode:    模式(train/valid)
            :param imgSize: 网络要求输入的图像尺寸

        Returns:
            precision, recall
        '''      
        self.contrast = contrast
        # 记录数据集大小
        self.dataSize = 0      
        # 数据集类别数      
        self.labelsNum = len(os.listdir(os.path.join(dir, mode)))           
        # 训练/验证 
        self.mode = mode              
        # 数据预处理方法
        self.tf = Transforms(imgSize=imgSize)
        # 遍历所有类别
        self.imgPathList, self.labelList = [], []
        '''对类进行排序，很重要!!!，否则会造成分类时标签匹配不上导致评估的精度很低(默认按字符串,如果类是数字还需要更改)'''
        catDirs = sorted(os.listdir(os.path.join(dir, mode)))
        for idx, cat in enumerate(catDirs):
            catPath = os.path.join(dir, mode, cat)
            labelFiles = os.listdir(catPath)
            # 每个类别里图像数
            length = len(labelFiles)
            # 存放图片路径
            self.imgPathList += [os.path.join(catPath, labelFiles[i]) for i in range(length)]
            # 存放图片对应的标签(根据所在文件夹划分)
            self.labelList += [idx for _ in range(length)]
            self.dataSize += length        


    def __getitem__(self, item):  
        '''重载data.Dataset父类方法, 获取数据集中数据内容
        '''   
        if self.contrast:
            img1, img2, img3, label = self.getContrastDataByIndex(item)
            return img1, img2, img3, label
        else:
            img, label = self.getDataByIndex(item)
            return img, label


    def getDataByIndex(self, item):
        # 读取图片
        img = Image.open(self.imgPathList[item]).convert('RGB')     
        img = np.array(img)
        # 获取image对应的label
        label = self.labelList[item]                 
        # 数据预处理/数据增强
        if self.mode=='train':
            img, _ = self.trainAlbumAug(img)
        if self.mode=='valid':
            img = self.normalAlbumAug(img)           
        return img.transpose(2,0,1), torch.LongTensor([label])
    


    def getContrastDataByIndex(self, item):
        # 读取图片
        img = Image.open(self.imgPathList[item]).convert('RGB')     
        img = np.array(img)
        # 获取image对应的label
        label = self.labelList[item]                 
        # 数据预处理/数据增强
        if self.mode=='train':
            # 原始图像增强(包含A.CoarseDrop)
            img_aug, trainTrans = self.trainAlbumAug(img)
            tranTrans2 = A.ReplayCompose.replay(trainTrans['replay'], image=img)
            # CLIP图像增强(不包含A.CoarseDrop, 其余和原始图像增强完全一致)
            img_clip, _ = self.trainAlbumAug(img, tranTrans2, drop_block=False)
            # 对比学习分支图像增强(包含A.CoarseDrop)
            img_contrast, _ = self.trainAlbumAug(img)
            return img_aug.transpose(2,0,1), img_clip.transpose(2,0,1), img_contrast.transpose(2,0,1), torch.LongTensor([label])
        
        if self.mode=='valid':       
            img = self.normalAlbumAug(img)    
            return img.transpose(2,0,1), torch.LongTensor([label])
    


    def trainAlbumAug(self, img, trainTrans=None, drop_block=True):
        """基于albumentations库的训练时数据增强
        """
        # albumentation的图像维度得是[W,H,C]
        if trainTrans==None:
            trainTrans = self.tf.trainTF(image=img)
        img = trainTrans['image']
        if drop_block:
            coarseDropTrans = self.tf.CoarseDropout(image=img)
            img = coarseDropTrans['image']
        img = self.normalAlbumAug(img)
        return img, trainTrans



    def normalAlbumAug(self, img):
        """基于albumentations库的基础数据预处理
        """
        norm_trans = self.tf.validTF(image=img)          
        img = norm_trans['image']   
        # 这里的box是coco格式(xywh)
        return img







    def __len__(self):
        '''重载data.Dataset父类方法, 返回数据集大小
        '''
        return self.dataSize
    
    def get_cls_num(self):
        '''返回数据集类别数
        '''
        return self.labelsNum







def visBatch(dataLoader, catNames):
    '''可视化训练集一个batch
    Args:
        dataLoader: torch的data.DataLoader
    Retuens:
        None     
    '''

    for step, batch in enumerate(dataLoader):
        # 只可视化一个batch的图像：
        if step > 0: break
        imgs = batch[0]
        labels = batch[1]
        # 图像均值
        mean = np.array([0.485, 0.456, 0.406]) 
        # 标准差
        std = np.array([[0.229, 0.224, 0.225]]) 
        plt.figure(figsize = (8,8))
        for idx, [img, label] in enumerate(zip(imgs, labels)):
            img = img.numpy().transpose((1,2,0))
            img = img * std + mean
            plt.subplot(8,8,idx+1)
            plt.imshow(img)
            plt.title(catNames[label], fontsize=8)
            plt.axis("off")
             # 微调行间距
            plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97, wspace=0.01, hspace=0.2)

        plt.savefig('./valid_data.jpg', dpi=200)









def visContrastBatch(dataLoader, catNames):
    '''可视化训练集一个batch
    Args:
        dataLoader: torch的data.DataLoader
    Retuens:
        None     
    '''

    for step, batch in enumerate(dataLoader):
        # 只可视化一个batch的图像：
        if step > 0: break
        imgs1, imgs2, imgs3 = batch[0], batch[1], batch[2]
        labels = batch[3]
        # 图像均值
        mean = np.array([0.485, 0.456, 0.406]) 
        # 标准差
        std = np.array([[0.229, 0.224, 0.225]]) 
        plt.figure(figsize = (8,8))
        for idx, [img, label] in enumerate(zip(imgs1, labels)):
            img = img.numpy().transpose((1,2,0))
            img = img * std + mean
            plt.subplot(8,8,idx+1)
            plt.imshow(img)
            plt.title(catNames[label], fontsize=8)
            plt.axis("off")
             # 微调行间距
            plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97, wspace=0.01, hspace=0.2)

        plt.savefig('./aug', dpi=200)
        plt.clf()

        for idx, [img, label] in enumerate(zip(imgs2, labels)):
            img = img.numpy().transpose((1,2,0))
            img = img * std + mean
            plt.subplot(8,8,idx+1)
            plt.imshow(img)
            plt.title(catNames[label], fontsize=8)
            plt.axis("off")
             # 微调行间距
            plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97, wspace=0.01, hspace=0.2)

        plt.savefig('./clip_aug', dpi=200)
        plt.clf()

        for idx, [img, label] in enumerate(zip(imgs3, labels)):
            img = img.numpy().transpose((1,2,0))
            img = img * std + mean
            plt.subplot(8,8,idx+1)
            plt.imshow(img)
            plt.title(catNames[label], fontsize=8)
            plt.axis("off")
             # 微调行间距
            plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97, wspace=0.01, hspace=0.2)

        plt.savefig('./contrast_aug', dpi=200)






# for test only
if __name__ == '__main__':
    datasetDir = 'E:/datasets/Classification/HUAWEI_cats_dogs_fine_grained/The_Oxford_IIIT_Pet_Dataset/images'
    mode = 'valid'
    bs = 64
    seed = 22
    img_size = [224, 224]
    seed_everything(seed)
    train_data = ClsDataset(datasetDir, mode, imgSize=img_size, contrast=False)
    print(f'数据集大小:{train_data.__len__()}')
    print(f'数据集类别数:{train_data.get_cls_num()}')
    train_data_loader = DataLoader(dataset = train_data, batch_size=bs, shuffle=True)
    # 获取label name
    catNames = sorted(os.listdir(os.path.join(datasetDir, mode)))
    # 可视化一个batch里的图像
    visBatch(train_data_loader, catNames)
    # visContrastBatch(train_data_loader, catNames)
    # 输出数据格式
    for step, batch in enumerate(train_data_loader):
        print(batch[0].shape)
        print(batch[1].shape)
        break