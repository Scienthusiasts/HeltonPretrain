import torch
import torch.nn as nn
from utils.utils import init_weights
# 注册机制
from register import MODELS
from utils.utils import NoSaveWrapper

@MODELS.register
class CLIPDistillEmbHead(nn.Module):
    '''Head
    '''
    def __init__(self, layers_dim:list, clip_model:nn.Module, distill_loss:nn.Module, contrast_loss:nn.Module, cat_names, template_prompt):
        '''网络初始化
            Args:
                layers_dim:     输入输出的维度大小, 例[C1, C2, ..., Cn]有n-1层Linear, 第i层维度是Ci+1
                clip_model:     CLIP模型
                distill_loss:   CLIP蒸馏损失
                contrast_loss:  CLIP图文对比损失
            Returns:
                None
        '''
        super(CLIPDistillEmbHead, self).__init__()
        self.cat_names = cat_names
        self.template_prompt = template_prompt
        self.cat_prompts = self._gen_cat_prompts()
        self.template_num = len(self.template_prompt)
        self.nc = len(self.cat_names)
        '''网络组件'''
        # 无论最后尺寸多大，都池化成1x1,这样输入的图像尺寸就可以是任意大小,但必须大于224x224
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = self.make_layers(layers_dim)
        # 蒸馏模型
        self._clip_model = NoSaveWrapper(clip_model.eval())
        self.init_cat_embs=False
        cat_embs = torch.zeros((self.nc, self.template_num, layers_dim[-1]))
        self.register_buffer('cat_embs', cat_embs)
        '''损失函数'''
        self.distill_loss = distill_loss
        self.contrast_loss = contrast_loss
        # 权重初始化
        init_weights(self.mlp, 'normal', 0.01)
        


    def make_layers(self, layers_dim):
        '''根据layers_dim生成自定义MLP层
        '''
        layers = []
        for i in range(len(layers_dim) - 1):
            in_dim = layers_dim[i]
            out_dim = layers_dim[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            # 最后一层只保留 Linear
            if i < len(layers_dim) - 2:  
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))
        # 参数为空时
        if len(layers) == 0:
            layers.append(nn.Identity())

        return nn.Sequential(*layers)



    def forward(self, x):
        """前向 
        Args:  
            x: 输入维度必须是[B, C, H, W]
        Returns:
            pred:  模型预测结果(logits, 未经过softmax)
        """
        x = self.gap(x).flatten(1)
        img_embs = self.mlp(x)
        # 图文计算余弦相似度, 相似度作为分类logits
        pred_logits = self._cosine_similarity(img_embs, self.cat_embs)

        return pred_logits



    def loss(self, x, y, img):
        """前向+计算损失 
        Args:  
            x:   输入维度必须是[B, C, h, w]
            y:   标签, [B]
            img: 原始图像, [B, 3, H, W]
        Returns:
            losses:          字典形式组织的损失
            img_text_logits: 预测logits
        """
        # 初始化self.cat_embs
        if self.init_cat_embs == False:
            self.cat_embs = self._clip_model(self.cat_prompts, type='text', device=x.device).reshape(self.nc, self.template_num, -1)
            self.init_cat_embs = True

        x = self.gap(x).flatten(1)
        img_embs = self.mlp(x)
        clip_img_embs = self._clip_model(img, type='image', device=x.device)
        rand_cat_embs = self._gen_train_cat_embs()
        img_text_logits = self._cosine_similarity(img_embs, rand_cat_embs, text_with_all_prompts=False)
        # 计算损失
        distill_loss = self.distill_loss(img_embs, clip_img_embs)
        contrast_loss = self.contrast_loss(img_text_logits, y)

        # 顺便计算并返回acc.指标
        pred_scores, pred_labels = torch.max(img_text_logits, dim=1)
        acc = sum(pred_labels==y) / y.shape[0]

        losses = dict(
            clip_distill_loss=distill_loss,
            clip_contrast_loss=contrast_loss,
            contrast_acc=acc
        )
        return losses


    def _gen_cat_prompts(self,):
        '''给定类别和prompt模板, 生成类别prompt(一次性生成所有组合)
        '''
        # 对于每个类别名, 根据给定的模板生成该类别的prompt
        cat_prompts = []
        for cat_name in self.cat_names:
            cat_prompts += [template.replace("{}", cat_name) for template in self.template_prompt]
        return cat_prompts


    def _gen_train_cat_embs(self,):
        '''生成训练时的随机类别prompt embeddings
            Returns:
                rand_cat_embs [nc, D]
        '''
        # 每个类别选一个prompt, 形状 [nc]
        rand_inds = torch.randint(0, self.template_num, (self.nc,))
        # [nc, dim]
        rand_cat_embs = self.cat_embs[torch.arange(self.nc), rand_inds, :]
        return rand_cat_embs
    

    # 计算图像和文本的余弦相似度
    def _cosine_similarity(self, img_f, text_f, scale=100., text_with_all_prompts=True):
        """计算图像和文本的余弦相似度
        Args: 
            img_f:  [B, D]
            text_f: [nc, N, D] / [nc, D]
        Returns:
            logits: [B, nc]

        """
        # 特征向量归一化
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        # 计算余弦相似度
        if text_with_all_prompts:
            logits = torch.einsum('bd,cnd->bnc', img_f, text_f)
            # 所有prompt的相似度ensemble [B, N, nc] -> [B, nc]
            logits = torch.mean(logits, dim=1)
        else:
            logits = torch.einsum('bd,cd->bc', img_f, text_f)
        return logits * scale
    






# for test only:
if __name__ == '__main__':
    template_prompt = [
            "a {} in the scene."
            "A {} with a happy expression.",
            "A photo of a friendly {}.",
            "A lovely little {}.",
            "a photo showing a {} in the scene",
            "a color picture of a {}, it is cute",
            "a photograph of a nice {}.",
            "a cropped photo of a {}, it is playful.",
            "I own a {} and I really like it.",
            "a picture of a {} taken long time ago.",
            "the picture showing a {} in the center.",
            "a picture of one {} in the scene.", # 
            "I adopted this {} several years ago.",
            "I took a picture of my {}.",
            "I love my {} and it loves me too.",
            "The {} in the picture is my friend's.",
            "This {} was a birthday present from my best friend.",
            "I accidentally snapped a picture of this {}.",
            "I petted my {} and it seemed to enjoy it.",
            "I called out to the {} and it rushed to me.",
            "My {} looking at the camera. It's the best memory ever.",
            "this {} used to be my best mate. Now it's gone.",
            "You're the best, my good {}.",
            "the {} is staring at me, want something to eat.",
            "My neighbour's {}, it looks funny.",
            "An elegant {} posing gracefully for the camera.",
            "This {} always greets me when I come home from work.",
            "The {} I rescued from the shelter last month.",
            "My {} waiting patiently for dinner time.",
            "The {} that helped me through difficult times.",
            "A {} communi{}ing through its expressive body language.",
            "this {}, A loyal companion with eyes full of devotion.",
            "A mischievous little {}, always getting into something new.",
            "Caught in the act! My {} staring guiltily at the mess it just made.",
            "my {}, The best welcome home committee, always waiting at the door.", 
            "My shadow, my {}, following me from room to room throughout the day.",
            "A blur of energy, the {} racing across the yard at full speed.",
            "A candid moment, this {} completely unaware of the camera.",
            "A calming influence, this {} knowing how to soothe a bad day.",
            "An unexpected friendship that grew into an unbreakable bond, my {}.",
            "The simple pleasure of watching my {} enjoy the sunshine.",
            "The first picture I took before we brought the {} home." 
        ]
    cat_names = ['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'Bengal', 'Birman', 
                'Bombay', 'boxer', 'British_Shorthair', 'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter', 
                'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'Maine_Coon', 
                'miniature_pinscher', 'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed', 
                'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']
    # model_cfgs = dict(
    #     type="ZeroShotCLIP",
    #     cat_names=cat_names, 
    #     template_prompt=template_prompt,
    #     clip_model=dict(
    #         type="Qihoo360FGCLIP",
    #         pretrain_path=pretrain_path
    #     )
    # )
    cls_loss = nn.CrossEntropyLoss()
    x = torch.randn(4, 2048, 7, 7)
    mlp = CLIPDistillEmbHead([2048, 1024, 768], cls_loss, cls_loss, cls_loss, cat_names, template_prompt)
    print(mlp)
    print(mlp._gen_cat_prompts())
    out = mlp(x)
    print(out.shape) 