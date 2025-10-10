from functools import partial
from einops import rearrange, reduce
import torch
from torch import nn, einsum
import torch.nn.functional as F

from generation.utils.utils import *
from generation.models.blocks import *
from utils.register import MODELS



@MODELS.register
class UNet(nn.Module):
    """DDPM 使用的 U-Net 网络结构
    ----------------------------------------
    结构特点：
      - Encoder-Decoder（U 型结构）
      - 每层包含两个 ResNet Block + Attention + 下/上采样层
      - 每个 ResBlock 内加入时间步嵌入（time embedding）
      - 中间层包含 Self-Attention
      - 支持 self-conditioning（将上一时刻预测的 x₀ 拼接输入）
    """

    def __init__(
        self,
        dim: int,
        init_dim: int = None,
        out_dim: int = None,
        dim_mults=(1, 2, 4, 8),
        channels: int = 3,
        self_condition: bool = False,
        resnet_block_groups: int = 4,
    ):
        super().__init__()

        # ========== 输入层与维度设定 ==========
        self.channels = channels
        self.self_condition = self_condition

        # 输入通道：是否使用自条件（self-conditioning）
        input_channels = channels * (2 if self_condition else 1)

        # 初始通道数
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, kernel_size=1)

        # 各层通道配置（如 64→128→256→512）
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # ========== 时间步嵌入模块 ==========
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),  # [B, dim]
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # 快捷构造 ResNet Block
        block = partial(ResnetBlock, groups=resnet_block_groups)

        # ========== 下采样路径（Encoder） ==========
        self.downs = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = (i == len(in_out) - 1)

            self.downs.append(
                nn.ModuleDict({
                    "block1": block(dim_in, dim_in, time_emb_dim=time_dim),
                    "block2": block(dim_in, dim_in, time_emb_dim=time_dim),
                    "attn": Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    "downsample": (
                        nn.Conv2d(dim_in, dim_out, 3, padding=1)
                        if is_last else Downsample(dim_in, dim_out)
                    ),
                })
            )

        # ========== 中间层（Bottleneck） ==========
        mid_dim = dims[-1]
        self.mid = nn.ModuleDict({
            "block1": block(mid_dim, mid_dim, time_emb_dim=time_dim),
            "attn": Residual(PreNorm(mid_dim, Attention(mid_dim))),
            "block2": block(mid_dim, mid_dim, time_emb_dim=time_dim),
        })

        # ========== 上采样路径（Decoder） ==========
        self.ups = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = (i == len(in_out) - 1)

            self.ups.append(
                nn.ModuleDict({
                    "block1": block(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    "block2": block(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    "attn": Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    "upsample": (
                        nn.Conv2d(dim_out, dim_in, 3, padding=1)
                        if is_last else Upsample(dim_out, dim_in)
                    ),
                })
            )

        # ========== 输出层 ==========
        self.out_dim = default(out_dim, channels)
        self.final = nn.Sequential(
            block(dim * 2, dim, time_emb_dim=time_dim),
            nn.Conv2d(dim, self.out_dim, 1),
        )

        # ========== 权重初始化 ==========
        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        """自定义权重初始化"""
        def init_fn(m):
            if isinstance(m, (nn.Conv2d, WeightStandardizedConv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(init_fn)

    # ------------------------------------------------------------------

    def forward(self, x, time, x_self_cond=None):
        """
        Args:
            x: [B, C, H, W] 输入图像（带噪声）
            time: [B] 当前扩散时间步
            x_self_cond: [B, C, H, W] 自条件输入（可选）
        """
        # 若使用 self-conditioning，则拼接上预测的 x₀
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        # 初始卷积
        x = self.init_conv(x)
        residual = x.clone()

        # 时间步嵌入
        t_emb = self.time_mlp(time)

        # -------- Encoder Path --------
        skip_connections = []
        for layer in self.downs:
            x = layer["block1"](x, t_emb)
            skip_connections.append(x)

            x = layer["block2"](x, t_emb)
            x = layer["attn"](x)
            skip_connections.append(x)

            x = layer["downsample"](x)

        # -------- Bottleneck --------
        x = self.mid["block1"](x, t_emb)
        x = self.mid["attn"](x)
        x = self.mid["block2"](x, t_emb)

        # -------- Decoder Path --------
        for layer in self.ups:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layer["block1"](x, t_emb)

            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layer["block2"](x, t_emb)
            x = layer["attn"](x)

            x = layer["upsample"](x)

        # -------- Final Output --------
        x = torch.cat((x, residual), dim=1)
        return self.final(x)




if __name__ == '__main__':
    model = UNet(
        dim=64,                 # 基础通道维度
        init_dim=None,          # 默认为 dim
        out_dim=3,              # 输出通道数（比如重建 RGB 图像）
        dim_mults=(1, 2, 4),    # 三层 U 结构
        channels=3,             # 输入通道 RGB
        self_condition=False,   # 不使用自条件
    )

    # ========== Step 2. 构造假输入 ==========
    B, C, H, W = 2, 3, 64, 64   # batch=2，64x64 RGB 图片
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))  # 时间步整数，可视为 timestep

    # ========== Step 3. 前向传播测试 ==========
    with torch.no_grad():
        y = model(x, t)
        print(f"输入: {x.shape} -> 输出: {y.shape}")

    # ========== Step 4. 反向传播测试 ==========
    y = model(x, t)
    loss = y.mean()
    loss.backward()
    print("✅ 前向和反向传播均成功！")


