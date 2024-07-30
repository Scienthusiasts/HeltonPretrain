import os                                   
import numpy as np
import torch
import torch.nn as nn
import importlib
from tqdm import tqdm, trange
import random
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch.nn.functional as F

from utils.clipUtils import COSSim
from utils.metricsUtils import *

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)




# 判断模型是否是pytorch的dp或ddp实例
def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)



def JSD(probA, probB):
    """
    计算批量 Jensen-Shannon 散度
    probA 和 probB 都是 [bs, C] 的概率分布矩阵
    返回 [bs, bs] 的 Jensen-Shannon 散度矩阵
    """
    # 扩展维度以进行广播计算
    probA = probA.unsqueeze(1)  # [bs, 1, C]
    probB = probB.unsqueeze(0)  # [1, bs, C]
    # 计算平均分布
    M = 0.5 * (probA + probB)   # [bs, bs, C]
    # 计算 KL 散度
    kl_pa_m = F.kl_div(probA.log(), M, reduction='none').sum(dim=2)  # [bs, bs]
    kl_pb_m = F.kl_div(probB.log(), M, reduction='none').sum(dim=2)  # [bs, bs]
    # 计算 Jensen-Shannon 散度
    js_div = 0.5 * (kl_pa_m + kl_pb_m)
    
    return js_div



def JSD_single(p, q):
    """
    计算单个样本对 (p, q) 的 Jensen-Shannon 散度
    p 和 q 应该是概率分布 (经过 softmax)
    """
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(p.log(), m, reduction='batchmean')
    kl_qm = F.kl_div(q.log(), m, reduction='batchmean')
    return 0.5 * (kl_pm + kl_qm)

def pairwiseJSD(tensorA, tensorB):
    """
    计算 tensorA 和 tensorB 的两两之间的 Jensen-Shannon 散度
    返回大小为 [bs, bs] 的散度矩阵
    """
    bs = tensorA.size(0)
    # 将向量转换为概率分布
    tensorA_prob = F.softmax(tensorA, dim=1)
    tensorB_prob = F.softmax(tensorB, dim=1)
    # 初始化 Jensen-Shannon 散度矩阵
    js_matrix = torch.zeros(bs, bs)
    # 计算两两之间的 Jensen-Shannon 散度
    for i in trange(bs):
        for j in range(bs):
            js_matrix[i, j] = JSD_single(tensorA_prob[i], tensorB_prob[j])
    
    return js_matrix






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
    from torchsummary import summary
    summary(model, input_size=(3,224,224))
    # torch.save(model.half().state_dict(), os.path.join("last_fp16.pt"))












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
    '''提取两个样本的embeddings'''
    pair_embeddings = []
    pair_images = []
    for img_path in img_pair_paths:
        image = Image.open(img_path).convert('RGB')
        # Image 转numpy
        image = np.array(image)
        pair_images.append(image)
        embedding = model.inferImgEmbedding(device, np.array(image), tf, half)
        pair_embeddings.append(embedding)
        

    '''计算相似度'''
    sim = COSSim(pair_embeddings[0], pair_embeddings[1]) / 100.
    '''计算动态阈值'''
    dynamic_T = model.head.lernable_T(torch.cat((pair_embeddings[0], pair_embeddings[1]), dim=1))
    dynamic_T = F.sigmoid(dynamic_T)
    # is_pair = sim.item() > dynamic_T.item()
    is_pair = sim.item() > 0.917

    '''可视化预测结果'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    fig.suptitle(f"These two samples are {['different', 'the same'][is_pair]} {['identities', 'identity'][is_pair]} | {'sim=%.2f, Td=%.2f'%(sim.item(), dynamic_T.item())}")
    # 在第一个子图中绘制图像
    ax1.set_title('sample 1')
    ax1.axis('off')
    ax1.imshow(pair_images[0])
    ax2.set_title('sample 2')
    ax2.axis('off')
    ax2.imshow(pair_images[1])
    # plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.90)
    plt.savefig('./id_res.jpg', dpi=200)

    print('图像对相似度: ', sim.item(), dynamic_T.item(), is_pair)
    return 





def IdentifyByDynamicT(model, device, tf, img_dir, half=False):
    model.eval()
    id_embeddings = []
    id_cats = []
    # 遍历个体文件夹
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
    # 所有样本两两相似度
    id_sim = COSSim(id_embeddings, id_embeddings).cpu().numpy() / 100.
    # 用js散度代替余弦相似度
    # id_jsd = pairwiseJSD(id_embeddings/100., id_embeddings/100.).cpu().numpy()
    # plt.imshow(id_jsd)
    # plt.show()
    # 生成标签
    id_labels = [np.where(id_cats==i)[0] for i in id_cats]
    # 生成GT矩阵
    target_M = np.zeros_like(id_sim).astype(bool)
    for i, id in enumerate(id_labels):
        target_M[i, id] = True
    # 总样本数
    sample_num = id_sim.shape[0]
    dynamic_T = np.zeros_like(id_sim)
    # 两两进行比较
    # for i in trange(sample_num):
    #     i_expanded = id_embeddings[i].expand(id_embeddings.size(0), -1)
    #     cat_embeddings = torch.cat((i_expanded, id_embeddings), dim=1)
    #     # 计算动态阈值
    #     dynamic_T[i] = F.sigmoid(model.head.lernable_T(cat_embeddings)).squeeze(1).detach().cpu().numpy()
    # 判别矩阵，如果相似度大于动态阈值，则认为是同一个体
    # judge_M = id_sim > dynamic_T
    judge_M = id_sim > 0.92
    # 获取对角线元素的索引
    indices = torch.arange(judge_M.shape[0])
    # 对角线元素不考虑在内(自己和自己比肯定很像, 意义不大)
    judge_M[indices, indices] = False
    target_M[indices, indices] = False
    # 判别矩阵和GT矩阵比较, 计算acc.
    TP = np.sum(judge_M & target_M)
    FN = np.sum(~judge_M & target_M)
    FP = np.sum(judge_M & ~target_M)
    TN = np.sum(~judge_M & ~target_M)
    acc = (TP + TN) / (TP + FP + TN + FN)
    pos_precision = TP / (TP + FP)
    neg_precision = TN / (TN + FN)
    pos_recall = TP / (TP + FN)
    neg_recall = TN / (TN + FP)
    print(f'acc={acc}\npos_precision={pos_precision}\nneg_precision={neg_precision}\npos_recall={pos_recall}\nneg_recall={neg_recall}')

        