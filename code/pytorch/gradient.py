import torch

# 自动求梯度

x = torch.tensor(1.0, requires_grad=True)               # 求导时需要设置 requires_grad=True
print(x)

y = 1.0 * torch.pow(x, 4) + 2.0 * x + 3.0
dy_dx = torch.autograd.grad(y, x, create_graph=True)    # 求高阶导时需要设置 create_graph=True
dy2_dx2 = torch.autograd.grad(dy_dx, x)
print(dy_dx)                                            # 输出是tuple
print(dy2_dx2)

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
