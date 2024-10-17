import torch

# 什么是梯度 - 就是偏导
# 模型的参数需要梯度，输入不需要梯度

# x是输入，不需要梯度；w、h是参数，需要梯度
x = torch.tensor(2.0)
print(x.requires_grad)
w = torch.tensor(10.0, requires_grad=True)
h = torch.tensor(1.0, requires_grad=True)
y = w * x + h
print(y)
print(y.requires_grad)
y.backward()
print(x.grad)
print(w.grad)
print(h.grad)

# 自动求梯度

x = torch.tensor(1.0, requires_grad=True)               # 求导时需要设置 requires_grad=True
print(x)

y = 1.0 * torch.pow(x, 4) + 2.0 * x + 3.0
dy_dx = torch.autograd.grad(y, x, create_graph=True)    # 求高阶导时需要设置 create_graph=True
dy2_dx2 = torch.autograd.grad(dy_dx, x)
print(dy_dx)                                            # 输出是tuple
print(dy2_dx2)

# 反向传播求所有 requires_grad=True 参数梯度

z = 2.0 * x ** 2
z.backward()
print(x.grad)

# 多个张量的梯度

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

# 定义函数 z = x^2 + y^3
z = x**2 + y**3

# 计算 z 对 x 和 y 的梯度
dz_dx, dz_dy = torch.autograd.grad(z, [x, y])

# 打印结果
print("dz/dx:", dz_dx)
print("dz/dy:", dz_dy)
