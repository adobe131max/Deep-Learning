import torch

# https://www.bilibili.com/video/BV15v411W78M

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
    multi_head_self_attention()
