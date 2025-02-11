import torch
import torch.nn as nn

# https://www.bilibili.com/video/BV15v411W78M

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        assert dim % heads == 0, 'dim % heads must == 0'
        self.dim = dim
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.o   = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor):
        # token_num, batch_size, token_dim
        N, B, D = x.shape
        assert D == self.dim, f'input token dim must equal {self.dim}'
        # [N, B, 3*D]
        qkv = self.qkv(x)
        # [N, B, 3, heads, dim_per_head]
        qkv = qkv.reshape(N, B, 3, self.heads, -1)
        # [3, B, heads, N, dim_per_head]
        qkv = qkv.permute(2, 1, 3, 0, 4)
        # [B, heads, N, dim_per_head]
        q, k, v = qkv
        # [B, heads, N, N]
        attn = q @ k.transpose(-2, -1)
        # scale weight
        attn *= self.scale
        # softmax line
        attn = attn.softmax(dim=-1)
        # [B, heads, N, dim_per_head]
        x = attn @ v
        # [B, N, heads, dim_per_head]
        x = x.transpose(1, 2)
        # [B, N, D]
        x = x.reshape(B, N, -1)
        # [B, N, D]
        x = self.o(x)
        return x


def self_attention():
    # (x1, x2, ..., xn)
    x = 'this is a sentence'

    # 1. embedding

    # x to a
    n = 100     # token数量
    d = 512     # 每个token的维度
    # (a1, a2, ..., an)
    a = torch.randn(n, d)

    # 2. 计算 Q、K、V (可以通过全连接实现，这里忽略了偏置)

    # 使用相同的 Wq、Wk、Wv
    Wq = torch.randn(d, d)
    Wk = torch.randn(d, d)
    Wv = torch.randn(d, d)

    Q = torch.mm(a, Wq)
    K = torch.mm(a, Wk)
    V = torch.mm(a, Wv)

    # 3, Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
    # dot-product
    r"""
    | 1,1 1,2 ... 1,n |
    | 2,1 2,1 ... 2,n |
    | ... ... ... ... |
    | n,1 n,2 ... n,n |
    """
    attentions = torch.mm(Q, K.t())
    # scale
    attentions /= torch.sqrt(torch.tensor(d))
    # row softmax
    attentions = torch.softmax(attentions, dim=1)
    print(attentions.shape)
    b = torch.mm(attentions, V)
    print(b.shape)


def multi_head_self_attention():
    head = 8
    n = 100
    d = 512
    dv = d // head      # 每个 head 的维度
    a = torch.randn(n, d)
    
    Wq = torch.randn(d, d)
    Wk = torch.randn(d, d)
    Wv = torch.randn(d, d)

    Q = torch.mm(a, Wq)
    K = torch.mm(a, Wk)
    V = torch.mm(a, Wv)
    
    # MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h)W^O
    # where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    # 实际的做法就是直接平均拆分了
    # W_i^Q, W_i^K, W_i^V
    WiQs = [torch.randn(d, dv) for _ in range(head)]
    WiKs = [torch.randn(d, dv) for _ in range(head)]
    WiVs = [torch.randn(d, dv) for _ in range(head)]
    
    # head_i
    bis = []
    for WiQ, WiK, WiV in zip(WiQs, WiKs, WiVs):
        Qi = torch.mm(Q, WiQ)
        Ki = torch.mm(K, WiK)
        Vi = torch.mm(V, WiV)
        
        attentions_i = torch.mm(Qi, Ki.t())
        attentions_i /= torch.sqrt(torch.tensor(dv))
        attentions_i = torch.softmax(attentions_i, dim=1)
        bi = torch.mm(attentions_i, Vi)
        
        bis.append(bi)
    print(bis[0].shape)
    
    b = torch.cat(bis, dim=1)
    print(b.shape)
    
    # shape: [head x d_v, d], but d_v = d / head, so shape is [d, d]
    Wo = torch.randn(d, d)
    b = torch.mm(b, Wo)
    print(b.shape)


if __name__ == '__main__':
    """
    self attention 不会改变 shape
    """
    # multi_head_self_attention()
    
    attention = Attention(512, 8)
    x = torch.randn(100, 4, 512)
    x = attention(x)
