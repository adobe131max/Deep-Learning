import torch

from torchvision.models import resnet50

model = resnet50()

path = './model.pth'

# 模型保存

# 1. 保存完整模型（包括结构与权重）
torch.save(model, path)
# 完整模型加载
model = torch.load(path)

# 2. 只保存模型权重
torch.save(model.state_dict(), path)
# 需要先创建模型
model = resnet50()
# 载入模型权重
model.load_state_dict(torch.load(path))
