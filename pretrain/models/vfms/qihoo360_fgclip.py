import torch.nn as nn
from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM
import math

from utils.utils import *
from pretrain.datasets.preprocess import Transforms
# 注册机制
from utils.register import MODELS



@MODELS.register
class Qihoo360FGCLIP(nn.Module):
    """FG-CLIP: Fine-Grained Visual and Textual Alignment: https://arxiv.org/abs/2505.05071
    """
    def __init__(self, pretrain_path):
        """初始化
            Args:
                pretrain_path: CLIP的权重路径
        """
        super(Qihoo360FGCLIP, self).__init__()
        # model
        self.clip_model = AutoModelForCausalLM.from_pretrained(pretrain_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        self.image_processor = AutoImageProcessor.from_pretrained(pretrain_path)
        self.walk_short_pos = True
        
        for param in self.clip_model.parameters():
            param.requires_grad_(False)

    def forward(self, x, type='image', device='cpu', *args, **kwargs):
        '''前向, 调用fgclip图像和文本编码器(API接口, 外部调用此方法)
        '''
        self.device = device
        if type == 'image':
            embs = self._forward_img(x)
        if type == 'image_dense':
            embs = self._forward_img_dense(x)
        if type == 'text':
            embs = self._forward_text(x)
        return embs


    def _forward_img(self, x):
        """前向, 调用fgclip图像编码器
            Args:
                x: [B, 3, H, W]
            Returns:
                img_embs: [B, dim]
        """
        with torch.no_grad():
            img_embs = self.clip_model.get_image_features(x)
            return img_embs
        

    def _forward_img_dense(self, x):
        """前向, 调用fgclip图像编码器, 得到dense特征图
            Args:
                x: [B, 3, H, W]
            Returns:
                img_embs: [B, dim, feat_h * feat_w]
        """
        with torch.no_grad():
            # [B, dim, feat_h * feat_w]
            dense_img_embs = self.clip_model.get_image_dense_features(x)
            return dense_img_embs
        

    def _forward_text(self, x):
        """前向, 调用fgclip文本编码器
            Args: 
                device: 当前使用的设备
                x: list[str1, ..., strB]
            Returns:
                text_embs: [B, dim]
        """
        with torch.no_grad():
            # 文本先得转tokens
            tokens = torch.tensor(self.tokenizer(x, max_length=77, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=self.device)
            text_embs = self.clip_model.get_text_features(tokens, walk_short_pos=self.walk_short_pos)
            return text_embs


        
    def forward_dense_heatmap(self, device, imgs, captions):
        """前向, 调用fgclip图像编码器, 生成dense_img_embs与text_embs的余弦相似度
            Args:
                imgs:     [B, 3, H, W]
                captions: list[str1, ..., strN]
            Returns:
                sim_heatmap: [B, feat_h, feat_w]
                patch_size:  feat_h=feat_w=patch_size
        """
        self.device = device
        with torch.no_grad():
            # [B, feat_h * feat_w, D]
            dense_img_embs = self._forward_img_dense(imgs)
            # [N, D]
            text_embs = self._forward_text(captions)
            # 余弦相似度 [B, feat_h * feat_w, D], [N, D] -> [B, feat_h * feat_w, N]
            text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
            dense_img_embs = dense_img_embs / dense_img_embs.norm(p=2, dim=-1, keepdim=True)
            sim_feat = torch.einsum('bcd,nd->bcn', dense_img_embs, text_embs)

            '''归一化'''
            # 范围归一化 [-1, 1] -> [0, 1]
            # sim_heatmap = (sim_feat + 1.) * 0.5
            # 最大最小归一化
            sim_max, sim_min = torch.max(sim_feat, dim=1, keepdim=True)[0], torch.min(sim_feat, dim=1, keepdim=True)[0]
            sim_heatmap = (sim_feat - sim_min) / (sim_max - sim_min)

            '''reshape'''
            # [B, feat_h * feat_w, N] -> [B, N, feat_h, feat_w]
            B, C, N = sim_heatmap.shape
            patch_size = int(math.sqrt(C))
            sim_heatmap = sim_heatmap.reshape(B, N, patch_size, patch_size) 
            return sim_heatmap, patch_size







# for test only:
if __name__ == '__main__':
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = [336, 336] # [224, 224]
    img_dir = '/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/train'
    base_ckpt_path = r'/mnt/yht/code/HeltonPretrain/ckpts/hugging_face/models--qihoo360--fg-clip-base/snapshots/f30d2b82ba939fd54ca732426f99f4d6c3c92387'
    large_ckpt_path = r'/mnt/yht/code/HeltonPretrain/ckpts/hugging_face/models--qihoo360--fg-clip-large/snapshots/19c2df7667052518ade09341652562b89b1332da'
    model = Qihoo360FGCLIP(large_ckpt_path).to(device)
    prompts = [
        "A cat with a happy expression.",
        "A photo of a friendly cat.",
        "A lovely little cat.",
        "a photo showing a cat in the scene",
        "a color picture of a cat, it is cute",
        "a photograph of a nice cat.",
        "a cropped photo of a cat, it is playful.",
        "I own a cat and I really like it.",
        "a picture of a cat taken long time ago.",
        "the picture showing a cat in the center.",
        "a picture of one cat in the scene.", # 
        "I adopted this cat several years ago.",
        "I took a picture of my cat.",
        "I love my cat and it loves me too.",
        "The cat in the picture is my friend's.",
        "This cat was a birthday present from my best friend.",
        "I accidentally snapped a picture of this cat.",
        "I petted my cat and she seemed to enjoy it.",
        "I called out to the cat and it rushed to me.",
        "My cat looking at the camera. It's the best memory ever.",
        "this cat used to be my best mate. Now it's gone.",
        "You're the best, my good cat.",
        "the cat is staring at me, want something to eat.",
        "My neighbour's cat, it looks funny.",
        "An elegant cat posing gracefully for the camera.",
        "This cat always greets me when I come home from work.",
        "The cat I rescued from the shelter last month.",
        "My cat waiting patiently for dinner time.",
        "The cat that helped me through difficult times.",
        "A cat communicating through its expressive body language.",
        "this cat, A loyal companion with eyes full of devotion.",
        "A mischievous little cat, always getting into something new.",
        "Caught in the act! My cat staring guiltily at the mess it just made.",
        "my cat, The best welcome home committee, always waiting at the door.", 
        "My shadow, following me from room to room throughout the day.",
        "A blur of energy, the cat racing across the yard at full speed.",
        "A candid moment, this cat completely unaware of the camera.",
        "A calming influence, this cat knowing how to soothe a bad day.",
        "An unexpected friendship that grew into an unbreakable bond, my cat.",
        "The simple pleasure of watching my cat enjoy the sunshine.",
        "The first picture I took before we brought the cat home." 
    ]
    img_dir = r'/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/valid'
    img_path = rf"{img_dir}/Maine_Coon/Maine_Coon_6.jpg"
    image = np.array(Image.open(img_path).convert('RGB'))

    # 图像预处理
    transform = Transforms(img_size)
    tensor_img = torch.tensor(transform.valid_transform(image=image)['image']).permute(2,0,1).unsqueeze(0).to(device)
    img_embs = model(device, tensor_img, modality='image')
    text_embs = model(device, prompts, modality='text')
    print(img_embs.shape, text_embs.shape)
    print(text_embs)
