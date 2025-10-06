import numpy as np
import torch
from PIL import Image, ImageFile
import torch.utils.data.dataset as data
from torch.utils.data import DataLoader
from functools import partial
import matplotlib.pyplot as plt
import os
import torch.distributed as dist
# 自定义
from utils.register import DATASETS
from utils.utils import seed_everything, worker_init_fn, natural_key
from pretrain.datasets.preprocess import Transforms
# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True







@DATASETS.register
class INDataset(data.Dataset):      
    '''有监督分类任务对应的基于ImageNet数据集组织格式的读取方式
    root
    ├─train
    │  ├─class_1
    │  ├─... ...
    │  └─class_n
    └─valid
        ├─class_1
        ├─... ...
        └─class_n
    '''
    def __init__(self, img_dir, mode, img_size, drop_block=True):    
        '''__init__() 为默认构造函数，传入数据集类别（训练或测试），以及数据集路径

        Args:
            img_dir:      图像数据集的根目录
            mode:     模式(train/valid)
            img_size: 网络要求输入的图像尺寸

        '''      
        # 训练时是否启用drop_block增强
        self.drop_block = drop_block
        # 记录数据集大小
        self.data_size = 0             
        # 训练/验证 
        assert mode in ('train', 'valid'), "mode must be 'train' or 'valid'"
        self.mode = mode              
        # 数据预处理方法
        self.transform = Transforms(img_size=img_size)
        # 遍历所有类别
        self.img_path_list, self.label_list = [], []
        '''对类进行排序，很重要!!!，否则会造成分类时标签匹配不上导致评估的精度很低'''
        # 只把目录当成类别(过滤文件、隐藏目录)
        all_entries = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d)) and not d.startswith('.')]
        self.cat_names = sorted(all_entries, key=natural_key)
        # 数据集类别数      
        self.labels_num = len(self.cat_names)   
        # 记录数据集所有图片的路径和对应的类别
        for idx, cat in enumerate(self.cat_names):
            cat_path = os.path.join(img_dir, cat)
            label_files = os.listdir(cat_path)
            # 每个类别里图像数
            length = len(label_files)
            # 存放图片路径
            self.img_path_list += [os.path.join(cat_path, label_files[i]) for i in range(length)]
            # 存放图片对应的标签(根据所在文件夹划分)
            self.label_list += [idx for _ in range(length)]
            self.data_size += length        
        
        # 打印数据集信息
        use_ddp = dist.is_initialized()
        if not use_ddp or use_ddp and dist.get_rank() == 0:
            print(f'dataset info: mode:{mode}, 图像数:{self.__len__()}, 类别数:{self.get_cls_num()}')


    def __getitem__(self, item):  
        '''重载data.Dataset父类方法, 获取数据集中数据内容
        '''   
        # 读取图片
        img = Image.open(self.img_path_list[item]).convert('RGB')     
        img = np.array(img)
        # 获取image对应的label
        label = self.label_list[item]                 
        # 数据预处理/数据增强
        if self.mode=='train':
            img, _ = self.train_aug(img)
        if self.mode=='valid':
            img = self.normal_aug(img)  

        return img.transpose(2,0,1), label
    

    def train_aug(self, img, train_transform=None):
        """训练时数据增强
        """
        # albumentation的图像维度得是[W,H,C]
        if train_transform==None:
            train_transform = self.transform.train_transform(image=img)
        img = train_transform['image']
        if self.drop_block:
            coarseDropTrans = self.transform.CoarseDropout(image=img)
            img = coarseDropTrans['image']
        img = self.normal_aug(img)
        return img, train_transform


    def normal_aug(self, img):
        """基础数据预处理
        """
        norm_trans = self.transform.valid_transform(image=img)          
        img = norm_trans['image']   
        return img

    def __len__(self):
        '''重载data.Dataset父类方法, 返回数据集大小
        '''
        return self.data_size
    
    def get_cls_num(self):
        '''返回数据集类别数
        '''
        return self.labels_num

    # DataLoader中collate_fn参数使用
    # 由于检测数据集每张图像上的目标数量不一
    # 因此需要自定义的如何组织一个batch里输出的内容
    def dataset_collate(self, batch):
        images, labels = [], []
        for img, label in batch:
            images.append(img)
            labels.append(label)
        # np -> tensor
        images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        labels  = torch.from_numpy(np.array(labels)).type(torch.LongTensor)
        return images, labels


    # for debug only:
    def _vis_INDataset_batch(self, epoch, step, batch, cat_names):
        '''可视化训练集一个batch
        Args:
            cat_names:   list, 类别名
        Retuens:
            None     
        '''
        # 图像均值 标准差
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([[0.229, 0.224, 0.225]]) 

        imgs = batch[0]
        labels = batch[1]
        plt.figure(figsize = (8,8))
        for idx, [img, label] in enumerate(zip(imgs, labels)):
            img = img.numpy().transpose((1,2,0))
            img = img * std + mean
            plt.subplot(8,8,idx+1)
            plt.imshow(img)
            plt.title(cat_names[label], fontsize=8)
            plt.axis("off")
            # 微调行间距
            plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97, wspace=0.01, hspace=0.2)

        plt.savefig(f'./epoch{epoch}_step{step}.jpg', dpi=300)








# for test only
if __name__ == '__main__':

    # 配置字典
    img_dir = r'/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/train'
    cfg = {
        "dataset_cfg": {
            "type": "INDataset",
            "img_dir": img_dir,
            "mode": "train",
            "img_size": [224, 224],
            "drop_block": False
        },
        "bs": 64,
        "seed": 42,
        "shuffle": True
    }

    dataset_cfg = cfg["dataset_cfg"]
    seed_everything(cfg["seed"])
    train_dataset = DATASETS.build_from_cfg(dataset_cfg)
    print(f'数据集大小:{train_dataset.__len__()}')
    print(f'数据集类别数:{train_dataset.get_cls_num()}')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=cfg["bs"], shuffle=cfg["shuffle"], num_workers=8, collate_fn=train_dataset.dataset_collate, worker_init_fn=partial(worker_init_fn, seed=cfg["seed"]))
    # 获取label name
    cat_names = sorted(os.listdir(img_dir))
    print(cat_names)
    # 输出数据格式
    for epoch in range(1, 10):
        for step, batch in enumerate(train_data_loader):
            print(batch[0].shape)
            print(batch[1].shape)
            if step == 0:
                # 可视化一个batch里的图像
                train_dataset._vis_INDataset_batch(epoch, step, batch, cat_names)