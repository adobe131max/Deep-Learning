import torch
from torch import nn    # 如果不太熟悉python可能会有疑惑：不是已经导入了touch吗？这样导入是为了直接使用nn，而不是torch.nn 等价于 import torch.nn as nn
import torch.nn.functional as F

net = nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,1))

X = torch.rand(size=(2, 4))
print(X)
output = net(X)
print(output)

# 访问参数
print(net[0].state_dict())  # 不难发现给定的神经元数量不包含偏置，会自动添加偏置
print(net[2].state_dict())

x = torch.randn((2, 3))
print(x)
print(F.relu(x))
