import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import clip

from utils.utils import *
from modules.datasets.preprocess import Transforms



def _gen_label_prompts(prompt:list[str], cat_ids:list[int], cat_names:list[str]):
    '''给定类别和prompt模板, 生成类别prompt
    '''
    # 对于每个类别名, 随机从给定的模板中选择一个生成该类别的prompt
    rand_id = np.random.randint(0, high=len(prompt), size=len(cat_ids))
    prompt_labels = [prompt[rand_id[i]].replace("/=/", cat_names[cat_ids[i]]) for i in range(len(cat_ids))]
    return prompt_labels



# 计算图像和文本的余弦相似度
def _cosine_similarity(img_f, text_f, scale=100.):
    """计算图像和文本的余弦相似度
    """
    # 特征向量归一化
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    text_f = text_f / text_f.norm(dim=-1, keepdim=True)
    # 计算余弦相似度
    logits = scale * img_f @ text_f.t()
    # return logits.softmax(dim=-1)
    return logits



class OpenAICLIP(nn.Module):
    """Learning Transferable Visual Models From Natural Language Supervision(CLIP): https://arxiv.org/abs/2103.00020
    """
    def __init__(self, device, img_size, cat_names, cat_prompts_train, cat_prompts_valid, pretrain_path):
        """初始化
            Args:
                device:            cuda/gpu
                img_size:          输入图像尺寸
                cat_names:         数据集类别名列表
                cat_prompts_train: 类别prompts模板(训练时,通常有多条)
                cat_prompts_valid: 类别prompts模板(推理时,通常是最有代表性的一条)
                pretrain_path:     CLIP的权重路径
        """
        super(OpenAICLIP, self).__init__()
        # 这里还有点奇怪, 如果不使用.to(self.device)就还是在cpu上推理
        self.device = device
        # 类别和类别数量
        self.cat_names = cat_names
        self.nc = len(self.cat_names)
        # model
        self.clip_model, self.preprocess = clip.load(pretrain_path, device=self.device)
        self.cat_prompts_train = cat_prompts_train
        self.cat_prompts_valid = cat_prompts_valid
        self.prompts_token_val = self.genValLabel()
        self.prompts_embeddings_val = self.clip_model.encode_text(self.prompts_token_val)
        # 图像预处理
        self.transform = Transforms(img_size)


    def genTrainLabel(self):
        '''将类别生成prompts向量(训练时,prompts可能不同)
        '''
        # 给定类别名称生成类别的prompt描述
        prompts_train = _gen_label_prompts(self.cat_prompts_train, [i for i in range(self.nc)], self.cat_names)
        # 将prompt描述向量化(tokenize)
        prompts_token_train = clip.tokenize(prompts_train).to(self.device)
        return prompts_token_train


    def genValLabel(self):
        '''将类别生成prompts向量(推理时,prompts只有一条)
        '''
        # 给定类别名称生成类别的prompt描述
        prompts_val = _gen_label_prompts(self.cat_prompts_valid, [i for i in range(self.nc)], self.cat_names)
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




    def batchVal(self, batch_datas):
        '''一个batch的推理
        '''
        batch_imgs, batch_labels = batch_datas[0].to(self.device), batch_datas[1].to(self.device).reshape(-1)
        # 1.直接调用openai-clip生成相似度(未归一化):
        # img_logits, text_logits = self.clip_model(batch_imgs, self.prompts_token_val)
        # 2.自己计算相似度(未归一化):
        image_embeddings, text_embeddings = self.forward(batch_imgs, self.prompts_token_val)
        img_logits = _cosine_similarity(image_embeddings, text_embeddings, scale=100.)

        # 预测结果对应置信最大的那个下标
        pred_label = torch.argmax(img_logits, dim=1)
        # 记录(真实标签true_list, 预测标签pred_list, 置信度soft_list)
        true = list(batch_labels.cpu().detach())
        pred = list(pred_label.cpu().detach())
        soft = list(np.array(img_logits.softmax(dim=-1).cpu().detach()))

        return pred, true, soft



    def infer(self, image:np.array, half=False):
        '''一张图像的推理
        '''
        tensor_img = torch.tensor(self.transform.valid_transform(image=image)['image']).permute(2,0,1).unsqueeze(0).to(self.device)
        if half: tensor_img = tensor_img.half()
        # 1.直接调用openai-clip生成相似度(未归一化):
        # img_logits, text_logits = self.clip_model(tensor_img, self.prompts_token_val)
        # 2.自己计算相似度(未归一化):
        image_embeddings, text_embeddings = self.forward(tensor_img, self.prompts_token_val)
        img_logits = _cosine_similarity(image_embeddings, text_embeddings)

        logits = img_logits.softmax(dim=-1).cpu().detach().numpy()[0]
        sorted_id = sorted(range(len(logits)), key=lambda k: logits[k], reverse=True)

        return logits, sorted_id
    



    def inferImgEmbedding(self, image:np.array, half=False):
        '''一张图像的推理
        '''
        tensor_img = torch.tensor(self.transform.valid_transform(image=image)['image']).permute(2,0,1).unsqueeze(0).to(self.device)
        if half: tensor_img = tensor_img.half()
        image_embeddings = self.forwardImg(tensor_img)
        print(image_embeddings.shape)

        return image_embeddings









# for test only:
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = [224, 224]
    img_dir = '/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/train'
    ckpt_path = "/mnt/yht/code/HeltonPretrain/ckpts/CLIP_ViT-B-32.pt"
    cat_prompts_train = [
        "A /=/ with a happy expression.",
        "A photo of a friendly /=/.",
        "A lovely little /=/.",
        "a photo showing a /=/ in the scene",
        "a color picture of a /=/, it is cute",
        "a photograph of a nice /=/.",
        "a cropped photo of a /=/, it is playful.",
        "I own a /=/ and I really like it.",
        "a picture of a /=/ taken long time ago.",
        "the picture showing a /=/ in the center.",
        "a picture of one /=/ in the scene.", # 
        "I adopted this /=/ several years ago.",
        "I took a picture of my /=/.",
        "I love my /=/ and it loves me too.",
        "The /=/ in the picture is my friend's.",
        "This /=/ was a birthday present from my best friend.",
        "I accidentally snapped a picture of this /=/.",
        "I petted my /=/ and she seemed to enjoy it.",
        "I called out to the /=/ and it rushed to me.",
        "My /=/ looking at the camera. It's the best memory ever.",
        "this /=/ used to be my best mate. Now it's gone.",
        "You're the best, my good /=/.",
        "the /=/ is staring at me, want something to eat.",
        "My neighbour's /=/, it looks funny.",
    ]
    cat_prompts_valid = [
        "a picture of one /=/ in the scene."
    ]
    cat_names = sorted(os.listdir(img_dir))
    model = OpenAICLIP(device, img_size, cat_names, cat_prompts_train, cat_prompts_valid, ckpt_path)

    img_dir = r'/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/valid'
    img_path = rf"{img_dir}/Maine_Coon/Maine_Coon_6.jpg"
    image = np.array(Image.open(img_path).convert('RGB'))
    logits, sorted_id = model.infer(image)
    print(cat_names[sorted_id[0]])
