import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.utils import init_weights
# 注册机制
from register import MODELS






class PositionalEncoding1D(nn.Module):
    """1D正余弦位置编码(1D版本, 适用于NLP或Flatten后的图像序列)
    """
    def __init__(self, dim:int, dropout:float=0.0, max_len:int=5000):
        """
            Args:
                dim:     模型词向量的维度(一个单词用几个维度表示)
                dropout: dropout比例
                max_len: 最大长度(一个句子里最多几个单词)
        """
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # [0,..., maxLen-1], shape=[max_len, 1]
        pos = torch.arange(0, max_len).unsqueeze(1)
        # div_term就是10000^(2i/dim), shape=[dim/2]
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))

        pe = torch.zeros(max_len, dim)
        # 偶数位置为正弦编码 PE(pos, 2i  ) = sin{pos / [10000^(2i/dim)]}
        pe[:, 0::2] = torch.sin(pos * div_term)
        # 奇数位置为余弦编码 PE(pos, 2i+1) = cos{pos / [10000^(2i/dim)]}
        pe[:, 1::2] = torch.cos(pos * div_term)
        # 多一个batchsize维度 [1, max_len, dim]
        self.pe = pe.unsqueeze(0)
        # 通过register_buffer注册过的张量会自动成为模型中的参数，随着模型移动(gpu/cpu)而移动，但是不会随着梯度进行更新
        # 缺点:保存模型权重时会额外保存这一份参数量
        # self.register_buffer("pe", pe)

    def forward(self, x:torch.Tensor):
        """
            Args:
                x: 输入的tokens, shape=[bs, seq_len, dim]
            Returns:
        """
        # shape x  = [bs, seq_len, vec_dim]
        # shape pe = [1 , seq_len, vec_dim]
        # 根据x的长度来截取pe, 使得pe的长度和x的长度一致
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)




class PositionalEncoding2D(nn.Module):
    """2D正余弦位置编码(用于对图像特征生成位置编码) 
    """
    def __init__(self, dim: int, dropout: float=0.0, max_h: int=128, max_w: int=128):
        """
        Args:
            dim:     特征维度 (必须是偶数, 最好还能被4整除)
            dropout: dropout比例
            max_h:   最大高度
            max_w:   最大宽度
        """
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("dim 必须能被4整除, 因为要分成sin/cos x 高/宽 四部分. got dim=%d" % dim)
        self.dropout = nn.Dropout(p=dropout)
        d_model_h = dim // 2
        d_model_w = dim // 2

        # pos indices
        pos_h = torch.arange(max_h).float().unsqueeze(1)  # [max_h,1]
        pos_w = torch.arange(max_w).float().unsqueeze(1)  # [max_w,1]
        # 每个维度的频率
        div_term_h = torch.exp(torch.arange(0, d_model_h, 2).float() * -(math.log(10000.0) / d_model_h))
        div_term_w = torch.exp(torch.arange(0, d_model_w, 2).float() * -(math.log(10000.0) / d_model_w))
        # H 方向编码
        pe_h = torch.zeros(max_h, d_model_h)   # [max_h, d_model_h]
        pe_h[:, 0::2] = torch.sin(pos_h * div_term_h)
        pe_h[:, 1::2] = torch.cos(pos_h * div_term_h)
        # W 方向编码
        pe_w = torch.zeros(max_w, d_model_w)   # [max_w, d_model_w]
        pe_w[:, 0::2] = torch.sin(pos_w * div_term_w)
        pe_w[:, 1::2] = torch.cos(pos_w * div_term_w)

        # combine to [max_h, max_w, dim]
        pe = torch.zeros(max_h, max_w, dim)
        pe[:, :, :d_model_h] = pe_h.unsqueeze(1)  # broadcast along W
        pe[:, :, d_model_h:] = pe_w.unsqueeze(0)  # broadcast along H
        self.pe = pe.unsqueeze(0)


    def forward(self, x: torch.Tensor, H: int, W: int):
        """
        Args:
            x: 输入 tokens, shape=[B, H*W, C]
            H, W: 高度和宽度
        """
        B, L, C = x.shape
        # slice, reshape and cast to same dtype/device as x
        pos_encoding = self.pe[:, :H, :W, :].reshape(1, H * W, C)
        return self.dropout(x + pos_encoding)







@MODELS.register
class ViTHead(nn.Module):
    '''Head
    '''
    def __init__(self, nc, in_dim, num_heads, cls_loss:nn.Module, mlp_ratio=4.0, dropout=0.0):
        '''网络初始化
            Args:
                nc:         分类类别数
                in_dim:     输入特征维度
                cls_loss:   分类损失实例
                mlp_ratio:  FFN隐藏层维度
                dropout:    FFN dropout ratio
            Returns:
                None
        '''
        super(ViTHead, self).__init__()

        '''网络结构'''
        # 正余弦位置编码
        self.pe = PositionalEncoding2D(in_dim)
        # 分类 token [1, 1, C]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_dim))  
        # LayerNorm + MultiheadAttention
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=False)
        # LayerNorm + FFN
        hidden_dim = int(in_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(in_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(dropout)
        )
        # 分类输出
        self.fc = nn.Linear(in_dim, nc)

        '''损失'''
        self.clsLoss = cls_loss

        # 权重初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        init_weights(self.attn, 'normal', 0, 0.01)
        init_weights(self.ffn, 'normal', 0, 0.01)



    def forward(self, x):
        '''前向传播
            Args:
                x: 输入维度必须是[B, C, H, W]
            Returns:
                logits: [B, nc]
            注意: 下面这两种写法不等价!, 第一种捕获到的关系更丰富
                attn_out, _ = self.attn(x_res, x_res, x_res)[:,0,:]
                attn_out, _ = self.attn(x_res[:,0,:], x_res, x_res)
        '''
        if self.pe.pe.device != x.device:
            self.pe.pe = self.pe.pe.to(x.device)
        '''得到输入特征'''
        B, C, H, W = x.shape
        # [B, C, H*W] -> [B, H*W, C]
        x = x.flatten(2).transpose(1, 2) 
        # 添加位置编码
        x = self.pe(x, H, W)
        # 分类 token expand [1, 1, C] -> [B, 1, C]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # 分类token与特征拼接, [B, H*W, C] -> [B, H*W+1, C]
        x = torch.cat([cls_tokens, x], dim=1)

        '''一层完整的 Transformer Encoder (ViT)'''
        # LN + MHA -> [B, H*W+1, C]
        x_res = self.norm1(x)
        attn_out, _ = self.attn(x_res, x_res, x_res)  
        x = x + attn_out
        # LN + FFN
        # 只取分类token做分类 (节省计算量的写法) [B, H*W+1, C] -> [B, C]
        x = x[:, 0, :]
        x_res = self.norm2(x)
        x = x + self.ffn(x_res)

        '''分类头'''
        # [B, C] -> [B, nc]
        logits = self.fc(x) 
        return logits
    

    def loss(self, x, y):
        '''前向传播+计算损失(训练时使用)
            Args:
                x: 输入维度必须是[B, C, H, W]
                y: 标签[B, ]
            Returns:
                losses: 字典形式组织的损失
        '''
        logits = self.forward(x)
        cls_loss = self.clsLoss(logits, y)
        # 顺便计算并返回acc.指标
        pred_logits, pred_labels = torch.max(logits, dim=1)
        acc = sum(pred_labels==y) / y.shape[0]
        # 组织成字典形式返回
        losses = dict(
            cls_loss = cls_loss,
            acc = acc
        )
        return losses











# for test only:
if __name__ == '__main__':
    cls_loss = nn.BCELoss()
    x = torch.randn(4, 2048, 7, 7)
    nc = 37
    head = ViTHead(nc, 2048, 8, cls_loss, mlp_ratio=1.0)
    print(head)
    out = head(x)
    print(out.shape) 