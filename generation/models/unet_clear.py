from functools import partial
from einops import rearrange, reduce
import torch
from torch import nn, einsum
import torch.nn.functional as F

from generation.utils.utils import *
from generation.models.blocks import *
from utils.register import MODELS





class Encoder(nn.Module):
    """UNet Encoder, 包含三个结构(初始层, 编码层, BottleNeck)
    """
    def __init__(self, input_dim, layer_dims, time_dim, resnet_block_groups=4):
        """
            Args:
                input_dim:           输入图像通道数(一般=3)
                layer_dims:          encoder通道数 例:[64, 64, 128, 256]
                time_dim:            时间编码维度
                resnet_block_groups: 每个残差block有几层卷积
            Returns:
                x:     最终输出特征
                skips: 中间的跨层特征用于后续残差拼接
        """
        super().__init__()
        res_block = partial(ResnetBlock, groups=resnet_block_groups)

        # init conv
        self.init_conv = nn.Conv2d(input_dim, layer_dims[0], 1)
        # downsample encoder
        in_out = list(zip(layer_dims[:-1], layer_dims[1:]))
        self.downs = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = (i == len(in_out) - 1)
            self.downs.append(nn.ModuleDict({
                "block1": res_block(dim_in, dim_in, time_emb_dim=time_dim),
                "block2": res_block(dim_in, dim_in, time_emb_dim=time_dim),
                "attn": Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                "downsample": Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            }))
        # BottleNeck
        self.mid_block1 = ResnetBlock(layer_dims[-1], layer_dims[-1], time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(layer_dims[-1], Attention(layer_dims[-1])))
        self.mid_block2 = ResnetBlock(layer_dims[-1], layer_dims[-1], time_emb_dim=time_dim)

    def forward(self, x, t):
        skips = []
        # init conv
        x = self.init_conv(x)
        skips.append(x)
        # encoder
        for layer in self.downs:
            x = layer['block1'](x, t)
            skips.append(x)
            x = layer['block2'](x, t)
            x = layer['attn'](x)
            skips.append(x)
            x = layer['downsample'](x)
        # bottle_neck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        return x, skips
    



class Decoder(nn.Module):
    """UNet Decoder, 包含三个结构(解码层, 输出层)
    """
    def __init__(self, output_dim, layer_dims, time_dim, resnet_block_groups=4):
        """
            Args:
                output_dim:          生成图像通道数(一般=3)
                layer_dims:          decoder通道数 例:[64, 64, 128, 256]
                time_dim:            时间编码维度
                resnet_block_groups: 每个残差block有几层卷积
            Returns:
                x: 最终生成的图像
        """
        super().__init__()
        res_block = partial(ResnetBlock, groups=resnet_block_groups)

        # decoder
        in_out = list(zip(layer_dims[:-1], layer_dims[1:]))
        self.ups = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = i == (len(in_out) - 1)
            self.ups.append(nn.ModuleDict({
                "block1": res_block(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                "block2": res_block(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                "attn": Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                "upsample": Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            }))
        # ouput conv
        self.final_res_block = ResnetBlock(layer_dims[0] * 2, layer_dims[0], time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(layer_dims[0], output_dim, 1)

    def forward(self, x, skips, t):
        # decoder
        for layer in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = layer['block1'](x, t)
            x = torch.cat((x, skips.pop()), dim=1)
            x = layer['block2'](x, t)
            x = layer['attn'](x)
            x = layer['upsample'](x)
        # ouput conv
        x = torch.cat((x, skips.pop()), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        return x





@MODELS.register
class UNet(nn.Module):
    """UNet(添加时序编码+线性注意力机制)
    """
    def __init__(self, input_dim, output_dim, layer_dims, resnet_block_groups=4):
        """
            Args:
                input_dim:           输入图像通道数(一般=3)
                output_dim:          生成图像通道数(一般=3)
                layer_dims:          decoder通道数 例:[64, 64, 128, 256]
                resnet_block_groups: 每个残差block有几层卷积
            Returns:
                x: 最终生成的图像
        """
        super().__init__()
        # 时间嵌入(DDPM需要知道step信息)
        time_dim = layer_dims[0] * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(layer_dims[0]),
            nn.Linear(layer_dims[0], time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        # Encoder
        self.encoder = Encoder(input_dim, layer_dims, time_dim, resnet_block_groups)
        # Decoder
        self.decoder = Decoder(output_dim, layer_dims, time_dim, resnet_block_groups)
        self.init_weights()


    def forward(self, x, time):
        # time embedding
        t = self.time_mlp(time)
        # 编码
        x, skips = self.encoder(x, t)
        # 解码
        x = self.decoder(x, skips, t)
        return x


    def init_weights(self):
        """权重初始化
        """
        def weight_init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d, WeightStandardizedConv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        self.apply(weight_init)















# ========== Debug 用例 ==========
if __name__ == '__main__':
    init_dim = 96
    model = UNet(
        input_dim=3,
        output_dim=3,
        # 配置 encoder / decoder 每一层的通道数
        layer_dims=[init_dim*1, init_dim*1, init_dim*2, init_dim*4],
    )
    # ========== Step 1. 统计模型参数量 ==========
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("🧩 模型参数统计：")
    print(f"  ➤ 总参数量：{total_params:,}")
    print(f"  ➤ 可训练参数量：{trainable_params:,}")
    print(f"  ➤ 参数占用显存约：{total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    # ========== Step 2. 构造输入 ==========
    B, C, H, W = 2, 3, 128, 128
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))

    # ========== Step 3. 前向传播 ==========
    with torch.no_grad():
        y = model(x, t)
        print(f"\n输入: {x.shape} -> 输出: {y.shape}")

    # ========== Step 4. 反向传播 ==========
    y = model(x, t)
    loss = y.mean()
    loss.backward()
    print("✅ 前向 + 反向传播成功！")