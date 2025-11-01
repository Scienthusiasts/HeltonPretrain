import numpy as np
import torch
from PIL import Image, ImageFile
import torch.utils.data.dataset as data
from torch.utils.data import DataLoader
from functools import partial
import matplotlib.pyplot as plt
import os
import torch.distributed as dist
# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 自定义
from heltonx.utils.register import DATASETS
from heltonx.utils.utils import seed_everything, worker_init_fn
from generation.datasets.preprocess import Transforms








@DATASETS.register
class GenDataset(data.Dataset):      
    '''有监督分类任务对应的数据集读取方式
    '''
    def __init__(self, img_dir, img_size):    
        '''__init__() 为默认构造函数，传入数据集类别（训练或测试），以及数据集路径

        Args:
            :param dir:      图像数据集的根目录
            :param mode:     模式(train/valid)
            :param img_size: 网络要求输入的图像尺寸

        Returns:
            precision, recall
        '''      
        self.img_dir = img_dir
        self.transform = Transforms(img_size=img_size)
        # 支持的图像扩展名
        IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        # 递归遍历所有子目录找到所有图像文件
        self.img_path_list = [
            os.path.join(root, fname)
            for root, _, files in os.walk(self.img_dir)
            for fname in files
            if fname.lower().endswith(IMG_EXTS)
        ]
        # 记录数据集大小
        self.dataSize = len(self.img_path_list)

        # 打印数据集信息
        use_ddp = dist.is_initialized()
        if not use_ddp or use_ddp and dist.get_rank() == 0:
            print(f'📄  dataset info: 图像数:{self.__len__()}')



    def __getitem__(self, item):  
        '''重载data.Dataset父类方法, 获取数据集中数据内容
        '''   
        # 读取图片
        img = Image.open(self.img_path_list[item]).convert('RGB')     
        img = np.array(img)
        # 数据增强
        img = self.albumAug(img)         
        return img.transpose(2,0,1)
    

    def albumAug(self, img):
        """基于albumentations库的基础数据预处理
        """
        trans = self.transform.transform(image=img)          
        img = trans['image']   
        return img


    def __len__(self):
        '''重载data.Dataset父类方法, 返回数据集大小
        '''
        return self.dataSize


    # DataLoader中collate_fn参数使用
    # 由于检测数据集每张图像上的目标数量不一
    # 因此需要自定义的如何组织一个batch里输出的内容
    def dataset_collate(self, batch):
        images = []
        for img in batch:
            images.append(img)
        # np -> tensor
        images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        return [images]
    

    # for debug only:
    def _vis_GenDataset_batch(self, epoch, step, batch):
        '''可视化训练集一个batch
        Args:
        Retuens:
            None     
        '''
        # 图像均值 标准差
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([[0.229, 0.224, 0.225]]) 

        imgs = batch
        plt.figure(figsize = (8,8))
        for idx, img in enumerate(imgs):
            img = img.numpy().transpose((1,2,0))
            img = img * std + mean
            plt.subplot(8,8,idx+1)
            plt.imshow(img)
            plt.axis("off")
            # 微调行间距
            plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97, wspace=0.01, hspace=0.2)

        plt.savefig(f'./epoch{epoch}_step{step}_.jpg', dpi=300)







# for test only
if __name__ == '__main__':

    # 配置字典
    img_dir = r'/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images'
    cfg = {
        "dataset_cfg": {
            "type": "GenDataset",
            "img_dir": img_dir,
            "img_size": [256, 256]
        },
        "bs": 64,
        "seed": 42,
        "shuffle": True
    }

    dataset_cfg = cfg["dataset_cfg"]
    seed_everything(cfg["seed"])
    train_dataset = DATASETS.build_from_cfg(dataset_cfg)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=cfg["bs"], shuffle=cfg["shuffle"], num_workers=8, collate_fn=train_dataset.dataset_collate, worker_init_fn=partial(worker_init_fn, seed=cfg["seed"]))
    # 输出数据格式
    for epoch in range(1, 10):
        for step, batch in enumerate(train_data_loader):
            print(batch.shape)
            if step == 0:
                # 可视化一个batch里的图像
                train_dataset._vis_GenDataset_batch(epoch, step, batch)