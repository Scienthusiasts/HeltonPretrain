import os                                   
import numpy as np
import torch
import torch.nn as nn
import importlib
from tqdm import tqdm
import random
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams

from utils.clipUtils import COSSim
from utils.metricsUtils import *

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)






def dynamic_import_class(module_path, class_name='module_name', get_class=True):
    '''动态导入类
    '''
    spec = importlib.util.spec_from_file_location(class_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if get_class:
        return getattr(module, class_name)
    else:
        return module

            



def loadWeightsBySizeMatching(model:nn.Module, ckpt_path:str):
    '''根据shape匹配导入权重
    '''
    print('Loading weights into state dict by size matching...')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(ckpt_path)
    a = {}
    for (kk, vv), (k, v) in zip(pretrained_dict.items(), model_dict.items()):
        try:    
            if np.shape(vv) ==  np.shape(v):
                # print(f'(previous){kk} -> (current){k}')
                a[k]=vv
        except:
            print(f'(previous){kk} mismatch (current){k}')
    model_dict.update(a)
    model.load_state_dict(model_dict)
    print('Finished!')

    return model






def init_weights(model, init_type, mean=0, std=0.01):
    '''权重初始化方法
    '''
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if init_type=='he':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if init_type=='normal':
                nn.init.normal_(module.weight, mean=mean, std=std)  # 使用高斯随机初始化
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)



def seed_everything(seed):
    '''设置全局种子
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False








def visInferResult(image:np.array, logits, sorted_id, cls_name, save_vis_path):
    # 超过10类则只显示top10的类别
    logits_top_10 = logits[sorted_id[:10]]
    cats_top_10 = [cls_name[i] for i in sorted_id[:10]] 
    '''CAM'''
    # # CAM需要网络能反传梯度, 否则会报错
    # # 要可视化网络哪一层的CAM(以mobilenetv3_large_100.ra_in1k为例, 不同的网络这部分还需更改)
    # target_layers = [self.model.backbone.blocks[-1]]
    # cam = GradCAM(model=self.model, target_layers=target_layers)
    # # 要关注的区域对应的类别
    # targets = [ClassifierOutputTarget(sorted_id[0])]
    # grayscale_cam = cam(input_tensor=img, targets=targets)[0].transpose(1,0)
    # visualization = show_cam_on_image(visImg / 255., grayscale_cam, use_rgb=True)
    '''可视化预测结果'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    # 在第一个子图中绘制图像
    ax1.set_title('image')
    ax1.axis('off')
    ax1.imshow(image)
    # ax1.imshow(visualization)
    # 在第二个子图中绘制置信度(横向)
    ax2.barh(cats_top_10, logits_top_10.reshape(-1))
    ax2.set_title('classification')
    ax2.set_xlabel('confidence')
    # 将数值最大的条块设置为不同颜色
    bar2 = ax2.patches[0]
    bar2.set_color('orange')
    # y轴上下反转，不然概率最大的在最下面
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.90)
    plt.savefig(save_vis_path, dpi=200)
    plt.clf() 






def inferenceSingleImg(model, device, tf, img_path, save_vis_path=False, half=False):
    '''推理一张图像
    '''
    image = Image.open(img_path).convert('RGB')
    # Image 转numpy
    image = np.array(image)
    '''推理一张图像'''
    cls_logits = model.infer(device, np.array(image), tf, half)
    sorted_id = sorted(range(len(cls_logits)), key=lambda k: cls_logits[k], reverse=True)
    '''是否可视化推理结果'''
    if save_vis_path:
        visInferResult(image, cls_logits, sorted_id, model.cls_name, save_vis_path)
    return cls_logits, sorted_id




def inferenceSingleImgTTA(model, device, tf, img_path, save_vis_path=False, half=False):
    '''推理一张图像(测试时增强, 一张原图一张镜像)
    '''
    image = Image.open(img_path).convert('RGB')
    # Image 转numpy
    img = np.array(image)
    flip_img = cv2.flip(img, 1)
    '''推理图像'''
    cls_logits = model.infer(device, np.array(img), tf, half)
    flip_cls_logits = model.infer(device, np.array(flip_img), tf, half)
    aug_logits = cls_logits * 0.5 + flip_cls_logits * 0.5

    sorted_id = sorted(range(len(aug_logits)), key=lambda k: cls_logits[k], reverse=True)
    '''是否可视化推理结果'''
    if save_vis_path:
        visInferResult(image, cls_logits, sorted_id, model.cls_name, save_vis_path)
    return cls_logits, sorted_id





def inferenceBatchImgs(model:nn.Module, device:str, tf, img_dir:str, cat_names:list[str], half=False, tta=False):
    '''推理图像s
    '''
    cat_names_dict = dict(zip(cat_names, [i for i in range(len(cat_names))]))
    # 记录(真实标签true_list, 预测标签pred_list, 置信度soft_list)
    true_list, pred_list, soft_list = [], [], []
    for cls_img_dir_name in tqdm(os.listdir(img_dir)):
        cls_img_dir = os.path.join(img_dir, cls_img_dir_name)
        for img_name in os.listdir(cls_img_dir):
            img_path = os.path.join(cls_img_dir, img_name)
            if tta:
                logits, sorted_id = inferenceSingleImgTTA(model, device, tf, img_path, save_vis_path=False, half=half)
            else:
                logits, sorted_id = inferenceSingleImg(model, device, tf, img_path, save_vis_path=False, half=half)
            soft_list.append(logits)
            pred_list.append(sorted_id[0])
            true_list.append(cat_names_dict[cls_img_dir_name])
            
    pred_list = np.array(pred_list)
    true_list = np.array(true_list)
    soft_list = np.array(soft_list)

    '''评估'''
    # 准确率
    acc = sum(pred_list==true_list) / pred_list.shape[0]
    # # 可视化混淆矩阵
    showComMatrix(true_list, pred_list, cat_names, './')
    # 绘制PR曲线
    PRs = drawPRCurve(cat_names, true_list, soft_list, './')
    # 计算每个类别的 AP, F1Score
    mAP, mF1Score, form = clacAP(PRs, cat_names)
    print('='*100)
    print(form)
    print('='*100)
    print(f"acc.: {acc} | mAP: {mAP} | mF1Score: {mF1Score}")
    print('='*100)










def Identify(model, device, tf, img_dir, half=False):
    id_embeddings = []
    id_cats = []
    avg_sim = []
    for id_set_dir in tqdm(os.listdir(img_dir)):

        id_img_dir = os.path.join(img_dir, id_set_dir)
        for id_img_name in os.listdir(id_img_dir):
            id_img_path = os.path.join(id_img_dir, id_img_name)
            image = Image.open(id_img_path).convert('RGB')
            # Image 转numpy
            image = np.array(image)
            '''推理一张图像'''
            img_embedding = model.inferImgEmbedding(device, np.array(image), tf, half)
            id_embeddings.append(img_embedding)
            id_cats.append(int(id_set_dir))
    id_cats = np.array(id_cats)
    # 计算相似度
    id_embeddings = torch.stack(id_embeddings, dim=1).squeeze(0)
    id_sim = COSSim(id_embeddings, id_embeddings)
    # 生成标签
    id_labels = [np.where(id_cats==i)[0] for i in id_cats]
    '''计算Recall'''
    Recall = 0
    for idx, id_label in enumerate(id_labels):
        print(id_sim[id_label])
        values, indices = id_sim[idx].topk(len(id_label), largest=True, sorted=True)
        # 将数组转换为集合
        indices = set(indices.cpu().numpy())
        id_label = set(id_label)
        # 计算交集
        intersection = indices.intersection(id_label)
        Recall += len(intersection) / len(id_label)

    Recall /= len(id_labels)
    print('recall:', Recall, len(id_labels))


    '''计算top1 为个体集合中任意匹配的召回率(除去本身)'''
    top1_recall = 0
    for idx, id_label in enumerate(id_labels):
        values, indices = id_sim[idx].topk(2, largest=True, sorted=True)[-1]
        # 将数组转换为集合
        indices = indices.item()
        id_label = set(id_label)
        if indices in id_label:
            top1_recall += 1

    top1_recall /= len(id_labels)
    print('top1:', top1_recall, len(id_labels))





def identifyPair(model, device, tf, img_pair_paths:list[str], half=False):
    '''anchor样本'''
    anchor_image = Image.open(img_pair_paths[0]).convert('RGB')
    # Image 转numpy
    anchor_image = np.array(anchor_image)
    # 推理
    anchor_embedding = model.inferImgEmbedding(device, np.array(anchor_image), tf, half)

    '''待识别样本'''
    unknown_image = Image.open(img_pair_paths[1]).convert('RGB')
    # Image 转numpy
    unknown_image = np.array(unknown_image)
    # unknown_image = cv2.flip(unknown_image, 1)

    # 推理
    unknown_embedding = model.inferImgEmbedding(device, np.array(unknown_image), tf, half)

    # 计算相似度
    sim = COSSim(anchor_embedding, unknown_embedding) / 100.
    print('图像对相似度: ', sim.item())