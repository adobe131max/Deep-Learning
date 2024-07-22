import torch
import torch.nn as nn
import torch.optim as optim
import math

# 定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim,
                 output_dim,
                 d_model=512,           # 嵌入维度
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,  # 编码器和解码器层中的前馈神经网络隐藏层的维度
                 dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_embedding = nn.Embedding(input_dim, d_model)   # 嵌入层：将每个整数 token 转换为一个高维向量
        self.tgt_embedding = nn.Embedding(output_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)          # 显式地加入位置编码，不会改变数据的维度
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        print(src.shape)
        print(src[0][0][:20])
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        print(src.shape)
        print(src[0][0][:20])
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        print(output.shape)
        output = self.fc_out(output)
        return output

# 示例：创建一个Transformer模型实例
input_dim = 1000    # 输入词汇表大小
output_dim = 1000   # 输出词汇表大小
src_seq_len = 32    # 源序列长度
tgt_seq_len = 32    # 目标序列长度

model = TransformerModel(input_dim, output_dim)

# 示例输入
# 每个token用一个整数表示，每列是一条完整输入
src = torch.randint(0, input_dim, (src_seq_len, 10))    # (source sequence length, batch size)
tgt = torch.randint(0, output_dim, (tgt_seq_len, 10))   # (target sequence length, batch size)
print(src)
print(src[:, 0])
print(src[:][0])

# 创建mask
src_mask = model.transformer.generate_square_subsequent_mask(src_seq_len)
tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_seq_len)

# 创建padding mask
src_padding_mask = torch.zeros((10, src_seq_len), dtype=torch.bool)         # (batch size, source sequence length)
tgt_padding_mask = torch.zeros((10, tgt_seq_len), dtype=torch.bool)         # (batch size, target sequence length)
memory_key_padding_mask = torch.zeros((10, src_seq_len), dtype=torch.bool)  # (batch size, source sequence length)

# 前向传播
output = model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
print(output.shape)  # 输出形状应为 (target sequence length, batch size, output dimension)
