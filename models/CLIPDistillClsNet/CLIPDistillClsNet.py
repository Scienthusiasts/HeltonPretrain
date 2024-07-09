import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

from utils.utils import *
from utils.clipUtils import COSSim
from models import CLIP
from models.CLIPDistillClsNet.Backbone import Backbone
from models.CLIPDistillClsNet.Head import Head
from models.CLIPDistillClsNet.DecoupledHead import DecoupledHead






class Model(nn.Module):
    '''完整FasterRCNN网络架构
    '''

    def __init__(self, cls_num, cls_name, backbone_name, loadckpt, backbone:dict, head:dict, clip:dict, infer_mode='cls'):
        super(Model, self).__init__()
        self.infer_mode = infer_mode
        self.cat_nums = cls_num
        self.cls_name = cls_name
        # Backbone最后一层输出的通道数
        input_c = {
            'vit_small_patch16_224.augreg_in21k_ft_in1k': 384,  #
            'tf_efficientnet_b4.ns_jft_in1k':             448,  # √
            'mobilenetv3_small_100.lamb_in1k':            576, 
            'mobilenetv3_large_100.ra_in1k':              960,
            'mobilenetv4_hybrid_medium.e500_r224_in1k':   960,  #
            'vit_base_patch16_224.augreg2_in21k_ft_in1k': 768, 
            'resnetaa50d.sw_in12k_ft_in1k':               2048, # √
            'efficientnet_b5.sw_in12k_ft_in1k':           2048,
            'resnetaa50d.d_in12k':                        2048, 
            'resnet50.a1_in1k':                           2048,
            'vgg16.tv_in1k':                              4096,
            'darknetaa53.c2ns_in1k':                      1024, 
            'cspdarknet53.ra_in1k':                       1024, 
            'cspresnext50.ra_in1k':                       2048,
            'efficientvit_m5.r224_in1k':                  384,
            }[backbone_name]
        
        '''网络组件'''
        # CLIPModel定义为全局变量, 而不是类成员
        global CLIPModel
        CLIPModel = CLIP.Model(backbone_name='ViT-B', **clip)
        # Backbone最好使用原来的预训练权重初始化
        self.backbone = Backbone(backbone_name=backbone_name, **backbone)
        self.head = Head(cls_num=self.cat_nums, input_c=input_c, clip_model=CLIPModel, **head)
        # self.head = DecoupledHead(cls_num=self.cat_nums, input_c=input_c, clip_model=CLIPModel, **head)

        '''是否导入预训练权重'''
        if loadckpt: 
            # self.load_state_dict(torch.load(loadckpt))
            # print('yolov5 pretrain ckpt loaded!')
            # 基于尺寸的匹配方式(能克服局部模块改名加载不了的问题)
            self = loadWeightsBySizeMatching(self, loadckpt)

    
    def forward(self, x):
        x = self.backbone(x)
        cls_logits, embeddings = self.head(x)
        # cls_logits, embeddings, contrast_embeddings = self.head(x)
        return cls_logits, embeddings
    


    def batchLoss(self, device, batch_datas):
        batch_imgs, batch_clip_imgs, batch_contrast_imgs, batch_labels = \
            batch_datas[0].to(device), \
            batch_datas[1].to(device), \
            batch_datas[2].to(device), \
            batch_datas[3].to(device).reshape(-1)
        # 将原始图像batch和对比学习图像batchcat在一起
        batch_combined_imgs = torch.cat([batch_imgs, batch_contrast_imgs], dim=0)
        batch_combined_labels = torch.cat([batch_labels, batch_labels], dim=0)
        # 提取图像特征
        backbone_combined_feat = self.backbone(batch_combined_imgs)
        loss = self.head.batchLoss(backbone_combined_feat, batch_clip_imgs, batch_combined_labels)
        return loss



    def batchVal(self, device, batch_datas):
        batch_imgs, batch_labels = batch_datas[0].to(device), batch_datas[1].to(device).reshape(-1)
        cls_logits, embeddings = self.forward(batch_imgs)
        # cls_logits, embeddings, contrast_embeddings = self.forward(batch_imgs)
        '''不同的推理模式'''
        output = self.inferByMode(cls_logits, embeddings)
        # 预测结果对应置信最大的那个下标
        pred_label = torch.argmax(output, dim=1)
        # 记录(真实标签true_list, 预测标签pred_list, 置信度soft_list)
        true = list(batch_labels.cpu().detach())
        pred = list(pred_label.cpu().detach())
        soft = list(np.array(output))

        return pred, true, soft




    def inferByMode(self, cls_logits, embeddings):
        '''不同的推理模式'''
        # 评估分类头的分类性能
        if self.infer_mode == 'cls':
            output = cls_logits.softmax(dim=-1).cpu().detach()
        # 评估embedding头的性能
        elif self.infer_mode == 'clip':
            img_logits = COSSim(embeddings, CLIPModel.prompts_embeddings_val.float())
            output = img_logits.softmax(dim=-1).cpu().detach()
        # 分类结果由分类头和embedding头共同决定
        elif self.infer_mode == 'ensemble':
            img_logits = COSSim(embeddings, CLIPModel.prompts_embeddings_val.float())
            logits = cls_logits.softmax(dim=-1) * 0.5 + img_logits.softmax(dim=-1) * 0.5
            output = logits.cpu().detach()

        return output





    def infer(self, device, image:np.array, tf, half=False):
        tensor_img = torch.tensor(tf.validTF(image=image)['image']).permute(2,0,1).unsqueeze(0).to(device)
        if half: tensor_img = tensor_img.half()
        with torch.no_grad():
            cls_logits, embeddings = self.forward(tensor_img)
            # cls_logits, embeddings, contrast_embeddings = self.forward(tensor_img)
        '''不同的推理模式'''
        output = self.inferByMode(cls_logits, embeddings)
        # 只有一张图片，所以返回[0]
        return output[0]




    def inferImgEmbedding(self, device, image:np.array, tf, half=False):
        '''一张图像的推理
        '''
        tensor_img = torch.tensor(tf.validTF(image=image)['image']).permute(2,0,1).unsqueeze(0).to(device)
        if half: tensor_img = tensor_img.half()
        with torch.no_grad():
            _, image_embeddings = self.forward(tensor_img)
            # _, image_embeddings, contrast_embeddings = self.forward(tensor_img)

        return image_embeddings
    


    def onnxInfer(self, onnx_model, device, image:np.array, tf):
        tensor_img = torch.tensor(tf.validTF(image=image)['image']).permute(2,0,1).unsqueeze(0)
        onnx_input_img = tensor_img.numpy()
        # ONNX Runtime 输出(输出的结果不是tensor, 而是numpy??)
        embeddings, cls_logits = onnx_model.run(['clip_head', 'cls_head'], {'input': onnx_input_img})
        cls_logits = torch.tensor(cls_logits).to(device)
        embeddings = torch.tensor(embeddings).to(device)
        '''不同的推理模式'''
        output = self.inferByMode(cls_logits, embeddings)[0]
        logits = output.numpy()
        sorted_id = sorted(range(len(logits)), key=lambda k: logits[k], reverse=True)

        return logits, sorted_id




















# for test only:
if __name__ == '__main__':

    cls_num = 101
    backbone_name='efficientvit_m5.r224_in1k'
    loadckpt = False
    backbone = dict(
        loadckpt = False,
        pretrain = False,
        froze = False
    )
    head = dict(
        mid_c = [256, 256, 256],
        kernel_s = [3,1,1]
    )
    cls_name = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 
                'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 
                'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 
                'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 
                'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 
                'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 
                'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 
                'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 
                'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 
                'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 
                'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']
    model = Model(cls_num, cls_name, backbone_name, loadckpt, backbone, head)
    torch.save(model.state_dict(), f"tmp.pt")

    # 验证 
    x = torch.rand((4, 3, 224, 224))
    out = model(x)
    print(out.shape)