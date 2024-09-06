import torch

print(torch.cuda.device_count())
print(torch.device('cpu'))
print(torch.device('cuda'))
print(torch.device('cuda:0'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.tensor([1,2,3])   # 默认在CPU上创建，储存在内存中
print(x.device)

X = torch.ones(2,3,device='cuda')
Y = torch.rand(2,3,device='cuda')
Z = X + Y
print(X)
print(Y)
print(Z)  # 需要在同一张GPU上才能运算
print(Z.device)

Xc = X.cpu()            # 复制一个，不改变原tensor
Xcpu = X.to('cpu')      # 复制一个，不改变原tensor
Xgpu = Xcpu.to('cuda')
print(X.device)
print(Xc.device)
print(Xcpu.device)
print(Xgpu.device)

print(id(X))
print(id(Xc))
print(id(Xcpu))
print(id(Xgpu))
