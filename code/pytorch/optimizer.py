import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

# https://blog.csdn.net/qyhaill/article/details/103043637

initial_lr = 0.2

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass

net_1 = model()
net_2 = model()

# optimizer

optimizer_1 = torch.optim.Adam(net_1.parameters(), lr = initial_lr)
print(optimizer_1.defaults)                 # dict 优化器的默认初始参数
print(len(optimizer_1.param_groups))        # list[dict] 如果没有为特定的参数组重新指定这些参数，将使用默认值
print(optimizer_1.param_groups[0].keys())
print(optimizer_1.param_groups[0]['lr'])

optimizer_2 = torch.optim.Adam([*net_1.parameters(), *net_2.parameters()], lr = initial_lr)
# optimizer_2 = torch.opotim.Adam(itertools.chain(net_1.parameters(), net_2.parameters())) # 和上一行作用相同
print(optimizer_2.defaults)
print(len(optimizer_2.param_groups))

optimizer_3 = torch.optim.Adam([{"params": net_1.parameters()}, {"params": net_2.parameters()}], lr = initial_lr)
print(optimizer_3.defaults)
print(len(optimizer_3.param_groups))

# lr_scheduler

# 这个epoch不需要传入，是根据scheduler_1.step()的数次从0开始
def f(epoch):
    print(f'step epoch: {epoch}')
    return 1 / (epoch + 2)

optimizer = torch.optim.Adam(net_1.parameters(), lr = initial_lr)
print(f"default learning rate: {optimizer.defaults['lr']}")
print(f"before create LRscheduler learning rate: {optimizer.param_groups[0]['lr']}")

scheduler = LambdaLR(optimizer, lr_lambda=f)                                        # 也可以写为： lr_lambda=lambda epoch: 1/(epoch+2)
print(f"after create LRscheduler learning rate: {optimizer.param_groups[0]['lr']}") # 0.2 → 0.1 说明在创建 lrscheduler 就已经调用了step()调整了一次 lr

# lr_scheduler更新optimizer的lr，是更新的optimizer.param_groups[n][‘lr’]，而不是optimizer.defaults[‘lr’]
for epoch in range(0, 10):
    print(f"epoch {epoch} learning rate: {optimizer.param_groups[0]['lr']}")
    for update in range(1):
        # output = model(input)             # 正向传播
        # loss = criterion(output, target)  # 计算 loss
        optimizer.zero_grad()               # 清楚上一轮的梯度
        # loss.backward()                   # 反向传播计算梯度
        optimizer.step()                    # 调整参数
    scheduler.step()                        # 更新 lr
    
print(optimizer.defaults['lr'])
