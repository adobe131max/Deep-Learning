import torch

# 1. 创建tensor

x = torch.Tensor([[1,2,3],[4,5,6]]) # 指定tensor内容
print(x)
print(x.dtype)                  # tensor中元素的类型

print(torch.Tensor(2,3))        # 创建tensor，内容未初始化
print(torch.IntTensor(2,3))     # 指定类型的tensor
print(torch.zeros(2,3))         # 全 0
print(torch.ones(2,3))          # 全 1
print(torch.randn(2,3))         # 浮点型随机数 - 均值为0，方差为1，服从b标准正态分布
print(torch.rand(2,3))          # 浮点型随机数 - [0, 1)
print(torch.arange(1,10))       # [start, end)

# 2. tensor运算

print(x.add(1))     # 生成一个新的tensor
print(x)
print(x.add_(1))    # 带_的直接在原tensor上改变
print(x)
print(x.view(-1))   # 改变形状
print(x)
print(x.zero_())    # 置 0

x = torch.Tensor([[1,2,3],[4,5,6]])
y = torch.tensor([[-1,-2,-3],[-4,-5,-6]])
print(torch.stack((x,y),0))
print(torch.stack((x,y),0).shape)
print(torch.stack((x,y),1))
print(torch.stack((x,y),1).shape)
print(torch.stack((x,y),2))
print(torch.stack((x,y),2).shape)

print(x+y)
print(x)
x+=y        # 直接改变原tensor
print(x)
