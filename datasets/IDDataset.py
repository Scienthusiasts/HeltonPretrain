import numpy as np
from PIL import Image, ImageFile
import torch.utils.data.dataset as data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import albumentations as A
import os
import torch
import random
# 自定义
from utils.utils import seed_everything
from datasets.preprocess import Transforms
# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True






class IDDataset(data.Dataset):      
    '''个体识别数据集读取方式
    '''
    def __init__(self, dir, imgSize):    
        '''__init__() 为默认构造函数，传入数据集类别（训练或测试），以及数据集路径

        Args:
            :param dir:     图像数据集的根目录
            :param mode:    模式(train/valid)
            :param imgSize: 网络要求输入的图像尺寸

        Returns:
            precision, recall
        '''      
        # 记录数据集大小
        self.dataSize = 0                    
        # 数据预处理方法
        self.tf = Transforms(imgSize=imgSize)
        self.id_dir = []
        '''遍历根目录(有三级目录), 生成id_dict'''
        # id_imgpath_dict的key是个体文件夹的路径, value是每个个体下的不同正样本(列表)
        # imgpath_list是图像路径, 方便__getitem__通过索引进行读取图像
        self.id_imgpath_dict = {}
        self.imgpath_list = []
        for dir1_name in os.listdir(dir):
            dir1 = os.path.join(dir, dir1_name)
            for id_dir_name in os.listdir(dir1):
                id_dir = os.path.join(dir1, id_dir_name)
                self.id_imgpath_dict[id_dir] = []
                for img_name in os.listdir(id_dir):
                    img_path = os.path.join(id_dir, img_name)
                    self.imgpath_list.append(img_path)
                    self.id_imgpath_dict[id_dir].append(img_path)
        self.data_size = len(self.imgpath_list)



    def __len__(self):
        '''重载data.Dataset父类方法, 返回数据集大小
        '''
        return self.data_size
    

    def __getitem__(self, item):  
        '''重载data.Dataset父类方法, 获取数据集中数据内容
        '''   
        anchor, pos, neg = self.getDataByIndex(item)
        return anchor, pos, neg


    def getDataByIndex(self, item):
        # 读取图片
        '''读取anchor'''
        anchor_path = self.imgpath_list[item]
        anchor = Image.open(anchor_path).convert('RGB')     
        anchor = np.array(anchor)  
        '''读取pos'''
        anchor_id_path = os.path.split(anchor_path)[0] 
        pos_list = self.id_imgpath_dict[anchor_id_path]
        # anchor样本在对应id文件夹中的位置索引   
        anchor_idx = pos_list.index(anchor_path)
        # 正样本随机在id文件夹中选择
        pos_idx = np.random.randint(len(pos_list))
        # 如果正样本取到anchor样本的位置, 则正样本直接取anchor样本的下一个元素
        if pos_idx==anchor_idx: pos_idx = ( anchor_idx + 1) % len(pos_list)
        pos_path = pos_list[pos_idx]
        pos = Image.open(pos_path).convert('RGB')     
        pos = np.array(pos)  
        '''读取neg'''
        # 随机到正样本的概率太小,就不做逻辑判断了
        neg_item = np.random.randint(self.data_size)
        neg_path = self.imgpath_list[neg_item]
        neg = Image.open(neg_path).convert('RGB')     
        neg = np.array(neg)  

        anchor = self.trainAlbumAug(anchor)
        pos = self.trainAlbumAug(pos)
        neg = self.trainAlbumAug(neg)

        return anchor.transpose(2,0,1), pos.transpose(2,0,1), neg.transpose(2,0,1)
    

    def trainAlbumAug(self, img):
        """基于albumentations库的训练时数据增强
        """
        # albumentation的图像维度得是[W,H,C]
        trainTrans = self.tf.idTF(image=img)
        img = trainTrans['image']
        img = self.normalAlbumAug(img)
        return img 


    def normalAlbumAug(self, img):
        """基于albumentations库的基础数据预处理
        """
        norm_trans = self.tf.validTF(image=img)          
        img = norm_trans['image']   
        # 这里的box是coco格式(xywh)
        return img


    # 设置Dataloader的种子
    # DataLoader中worker_init_fn参数使
    # 为每个 worker 设置了一个基于初始种子和 worker ID 的独特的随机种子, 这样每个 worker 将产生不同的随机数序列，从而有助于数据加载过程的随机性和多样性
    @staticmethod
    def worker_init_fn(worker_id, seed, rank=0):
        worker_seed = rank + seed
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)








def visBatch(batch):
    '''可视化训练集一个batch
    Args:
        dataLoader: torch的data.DataLoader
    Retuens:
        None     
    '''

    # 只可视化一个batch的图像：
    anchor = batch[0]
    pos = batch[1]
    neg = batch[2]
    # 图像均值
    mean = np.array([0.485, 0.456, 0.406]) 
    # 标准差
    std = np.array([[0.229, 0.224, 0.225]]) 
    plt.figure(figsize = (8,8))
    for idx, img in enumerate(anchor):
        img = img.numpy().transpose((1,2,0))
        img = img * std + mean
        plt.subplot(8,8,idx+1)
        plt.imshow(img)
        plt.axis("off")
            # 微调行间距
        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97, wspace=0.01, hspace=0.2)
    plt.savefig('./anchor.jpg', dpi=200)

    for idx, img in enumerate(pos):
        img = img.numpy().transpose((1,2,0))
        img = img * std + mean
        plt.subplot(8,8,idx+1)
        plt.imshow(img)
        plt.axis("off")
            # 微调行间距
        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97, wspace=0.01, hspace=0.2)
    plt.savefig('./pos.jpg', dpi=200)

    for idx, img in enumerate(neg):
        img = img.numpy().transpose((1,2,0))
        img = img * std + mean
        plt.subplot(8,8,idx+1)
        plt.imshow(img)
        plt.axis("off")
            # 微调行间距
        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97, wspace=0.01, hspace=0.2)
    plt.savefig('./neg.jpg', dpi=200)






# for test only
if __name__ == '__main__':
    datasetDir = 'E:/datasets/Classification/HUAWEI_cats_dogs_fine_grained/DogFace/after_4_bis'
    bs = 64
    seed = 22
    img_size = [224, 224]
    seed_everything(seed)
    train_data = IDDataset(datasetDir, imgSize=img_size)
    print(f'数据集大小:{train_data.__len__()}')
    train_data_loader = DataLoader(dataset = train_data, batch_size=bs, shuffle=True)
    # 输出数据格式
    train_batch = iter(train_data_loader)
    batch = next(train_batch)
    # 可视化一个batch里的图像
    visBatch(batch)
    print(batch[0].shape)
    print(batch[1].shape)
    print(batch[2].shape)