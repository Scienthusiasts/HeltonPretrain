import torch.nn as nn
import timm
import math
import torch.nn.functional as F

from utils.utils import *
from modules.datasets.preprocess import Transforms
# 注册机制
from register import MODELS



@MODELS.register
class DINOv3(nn.Module):
    """DINOv3(支持多分辨率输入, 分辨率是16的整数倍)
    """
    def __init__(self, model_name: str, pretrained=True, load_ckpt=None):
        """
        Args:
            model_name:     str, timm 中模型名称
            pretrained:     是否加载预训练权重(官方的权重)
            features_only:  是否去掉分类头，仅保留特征提取层
            out_layers:     list[int] 或 None, 指定哪些 stage 层输出特征
            froze_backbone: 是否冻结骨干网络
            load_ckpt:      是否加载本地预训练权重(自定义权重)
        """
        super().__init__()
        # features_only=True 直接去掉分类头
        self.dinov3 = timm.create_model(model_name, pretrained=pretrained)
        # 是否导入权重:
        if load_ckpt:
            self.dinov3 = load_state_dict_with_prefix(self.dinov3, load_ckpt, prefixes_to_try=['model.'])


        self.dinov3.eval()


    def forward(self, x):
        '''前向
        '''
        with torch.no_grad():
            x = self.dinov3(x) 
        return x


    def forward_dense(self, x):
        '''前向
        '''
        B, C, H, W = x.shape
        with torch.no_grad():
            x = self.dinov3.forward_features(x) 
            special_tokens, feature_map = self.split_vit_output(x, H//16, W//16)
        return special_tokens, feature_map


        
    def split_vit_output(self, x, h=16, w=16, num_extra_tokens=5):
        """
        将ViT输出 (BS, h*w+5, 1280) 拆分成 特殊tokens + 特征图
        参数:
            x: torch.Tensor, 形状 (BS, h*w+5, 1280)
            h, w: patch 网格大小 (默认16x16)
            num_extra_tokens: 特殊tokens数量 

        返回:
            special_tokens: (BS, num_extra_tokens, C)
            feature_map: (BS, C, H, W)
        """
        B, N, C = x.shape
        assert N == h * w + num_extra_tokens, f"输入序列长度 {N} 不等于 {h*w + num_extra_tokens}"
        # 前 num_extra_tokens 是 cls + register tokens
        special_tokens = x[:, :num_extra_tokens, :]  # (BS, 65, 1280)
        # 后面是 patch tokens，reshape 成 feature map
        patch_tokens = x[:, num_extra_tokens:, :]  # (BS, 256, 1280)
        feature_map = patch_tokens.transpose(1, 2).reshape(B, C, h, w)  # (BS, 1280, 16, 16)

        return special_tokens, feature_map



    def cosine_similarity_map(self, x, row: int, col: int) -> torch.Tensor:
        """
        给定特征图和一个像素位置，计算该像素 embedding 与所有位置 embedding 的余弦相似度。
        参数:
            fmap: torch.Tensor, 形状 [B, C, H, W]，特征图
            row: int, 行索引 (0 <= row < H)
            col: int, 列索引 (0 <= col < W)

        返回:
            sim_map: torch.Tensor, 形状 [B, H, W]，相似度特征图
        """
        B, C, H, W = x.shape
        assert 0 <= row < H and 0 <= col < W, f"row={row}, col={col} 超出范围 H={H}, W={W}"
        # 取出目标位置的 embedding，形状 [B, C]
        target = x[:, :, row, col]  
        # 归一化 target 和 x
        target_norm = F.normalize(target, dim=-1)  # (B, C)
        x_norm = F.normalize(x.view(B, C, -1), dim=1)  # (B, C, H*W)
        # 计算余弦相似度: [B, C] x [B, C, H*W] -> [B, H*W]
        sim = torch.bmm(target_norm.unsqueeze(1), x_norm).squeeze(1)  
        # reshape 回特征图
        sim_map = sim.view(B, H, W)

        return sim_map






# for test only:
if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = [1024, 1024] 
    img_dir = '/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/train'
    # model = DINOv3("vit_huge_plus_patch16_dinov3.lvd1689m", pretrained=False, load_ckpt='ckpts/vit_huge_plus_patch16_dinov3.lvd1689m.pt').to(device)
    model = DINOv3("vit_small_patch16_dinov3.lvd1689m", pretrained=False, load_ckpt='ckpts/backbone_vit_small_patch16_dinov3.lvd1689m.pt').to(device)
    print(model)
    img_dir = r'/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/valid'
    img_path = rf"{img_dir}/Maine_Coon/Maine_Coon_41.jpg"
    image = np.array(Image.open(img_path).convert('RGB'))

    # 图像预处理
    transform = Transforms(img_size)
    tensor_img = torch.tensor(transform.valid_transform(image=image)['image']).permute(2,0,1).unsqueeze(0).to(device)
    special_tokens, feature_map = model.forward_dense(tensor_img)
    # 与指定位置的注意力图
    row, col = 21,21
    heatmap = model.cosine_similarity_map(feature_map, row, col).squeeze(0).cpu().numpy()
    plt.imshow(heatmap)
    plt.text(
        col, row, "+",
        color="red", fontsize=14, fontweight="bold",
        ha="center", va="center"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"infer_result1.png", dpi=200)


