'''
tensorboard 记录训练指标，可视化神经网络训练运行的结果
启动：
1. 进入conda对应的环境
2. tensorboard --logdir 日志所在目录
3. 浏览器访问: http://localhost:6006/
'''
import time

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

for epoch in range(10):
    writer.add_scalar('linear', epoch, epoch)     # graph_name data step
    writer.add_scalar('quadratic', epoch**2, epoch)
    writer.add_scalar('cubic', epoch**3, epoch)
    
time.sleep(1)   # 如果结束的太快可能会丢失一些记录
