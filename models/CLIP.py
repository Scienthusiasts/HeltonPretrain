import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import clip

from utils.utils import *
from utils.clipUtils import genLabel, COSSim






class Model(nn.Module):
    '''完整FasterRCNN网络架构
    '''
    # backbone_name no used, but must kept
    def __init__(self, device, backbone_name, cls_names, prompts_template_train, prompts_template_val, weight_path):
        super(Model, self).__init__()
        # 这里还有点奇怪, 如果不使用.to(self.device)就还是在cpu上推理
        self.device = device
        # 类别
        self.cls_name = cls_names
        self.cls_num = len(self.cls_name)
        # model
        self.clip_model, self.preprocess = clip.load(weight_path, device=self.device)
        self.prompts_template_train = prompts_template_train
        self.prompts_template_val = prompts_template_val
        self.prompts_token_val = self.genValLabel()
        self.prompts_embeddings_val = self.clip_model.encode_text(self.prompts_token_val)


    def genTrainLabel(self):
        '''将类别生成prompts向量(训练时,prompts可能不同)
        '''
        # 给定类别名称生成类别的prompt描述
        prompts_train = genLabel(self.prompts_template_train, [i for i in range(self.cls_num)], self.cls_name)
        # 将prompt描述向量化(tokenize)
        prompts_token_train = clip.tokenize(prompts_train).to(self.device)
        return prompts_token_train


    def genValLabel(self):
        '''将类别生成prompts向量(推理时,prompts只有一条)
        '''
        # 给定类别名称生成类别的prompt描述
        prompts_val = genLabel(self.prompts_template_val, [i for i in range(self.cls_num)], self.cls_name)
        # 将prompt描述向量化(tokenize)
        prompts_token_val = clip.tokenize(prompts_val).to(self.device)
        return prompts_token_val
    

    def forwardImg(self, img_f):
        '''前向, 调用openai-clip图像编码器
        '''
        with torch.no_grad():
            image_embeddings = self.clip_model.encode_image(img_f)
            return image_embeddings
        

    def forwardText(self, text_f):
        '''前向, 调用openai-clip文本编码器
        '''
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_f)
            return text_embeddings


    def forward(self, img_f, text_f):
        '''前向, 调用openai-clip图像和文本编码器
        '''
        with torch.no_grad():
            image_embeddings = self.clip_model.encode_image(img_f)
            text_embeddings = self.clip_model.encode_text(text_f)
            return image_embeddings, text_embeddings




    def batchVal(self, device, batch_datas):
        '''一个batch的推理
        '''
        batch_imgs, batch_labels = batch_datas[0].to(device), batch_datas[1].to(device).reshape(-1)
        # 1.直接调用openai-clip生成相似度(未归一化):
        # img_logits, text_logits = self.clip_model(batch_imgs, self.prompts_token_val)
        # 2.自己计算相似度(未归一化):
        image_embeddings, text_embeddings = self.forward(batch_imgs, self.prompts_token_val)
        img_logits = COSSim(image_embeddings, text_embeddings)

        # 预测结果对应置信最大的那个下标
        pred_label = torch.argmax(img_logits, dim=1)
        # 记录(真实标签true_list, 预测标签pred_list, 置信度soft_list)
        true = list(batch_labels.cpu().detach())
        pred = list(pred_label.cpu().detach())
        soft = list(np.array(img_logits.softmax(dim=-1).cpu().detach()))

        return pred, true, soft



    def infer(self, device, image:np.array, tf, half=False):
        '''一张图像的推理
        '''
        tensor_img = torch.tensor(tf.validTF(image=image)['image']).permute(2,0,1).unsqueeze(0).to(device)
        if half: tensor_img = tensor_img.half()
        # 1.直接调用openai-clip生成相似度(未归一化):
        # img_logits, text_logits = self.clip_model(tensor_img, self.prompts_token_val)
        # 2.自己计算相似度(未归一化):
        image_embeddings, text_embeddings = self.forward(tensor_img, self.prompts_token_val)
        img_logits = COSSim(image_embeddings, text_embeddings)

        logits = img_logits.softmax(dim=-1).cpu().detach().numpy()[0]
        sorted_id = sorted(range(len(logits)), key=lambda k: logits[k], reverse=True)

        return logits, sorted_id
    



    def inferImgEmbedding(self, device, image:np.array, tf, half=False):
        '''一张图像的推理
        '''
        tensor_img = torch.tensor(tf.validTF(image=image)['image']).permute(2,0,1).unsqueeze(0).to(device)
        if half: tensor_img = tensor_img.half()
        image_embeddings = self.forwardImg(tensor_img)
        print(image_embeddings.shape)

        return image_embeddings









# for test only:
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_cat_dir = 'E:/datasets/Classification/food-101/images/train'
    weight_path = "F:/DeskTop/git/HeltonPretrain/ckpt/CLIP_ViT-B-32.pt"
    prompts_template_train = [
        "a picture of /=/, a kind of food.",
        "a picture of one /=/ in the scene, a sort of food that seems delicious.",
        "a photo showing the /=/ in the center, a kind of food.",
        "there is a /=/ in the scence, a kind of food.",
        "a color picture of a /=/, a kind of food.",
        "a photograph of a nice /=/, a sort of food.",
        "a photograph of a nice /=/ I took recently, delicious!",
        "a cropped photo of a /=/, a kind of food.",
        "I made a /=/ and I really like it.",
        "a picture of a /=/ taken long time ago, a sort of food.",
        "A picture of /=/ that seems delicious.",
        "A picture of food, the category of which is /=/.",
        "a food in the scence, it is /=/.",
        "I made this /=/, a kind of food, which is tasty.",
    ]
    prompts_template_val = [
        "a picture of one /=/ in the scene, a sort of food that seems delicious"
    ]
    model = Model(img_cat_dir, prompts_template_train, prompts_template_val, weight_path)
