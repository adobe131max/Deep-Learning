import torch

print(torch.cuda.device_count())
print(torch.device('cpu'))
print(torch.device('cuda'))
print(torch.device('cuda:0'))

x = torch.tensor([1,2,3])   # 默认在CPU上创建，储存在内存中
print(x.device)

X = torch.ones(2,3,device='cuda')
Y = torch.rand(2,3,device='cuda')
print(X)
print(Y)
print(X+Y)  # 需要在同一张GPU上才能运算
