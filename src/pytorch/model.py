import torch
import torch.nn as nn

from torchvision.models import resnet50

def save_and_load():
    path = './model.pth'

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

model = resnet50()

# 模型的下一层子结构
for name, module in model.named_children():
    print(name)

# 模型的所有参数
for name, param in model.named_parameters():
    print(name)

# nn.Suquential
# 模块之间的连接需要在forward方法中显式地指定，更加灵活，可以根据不同的输入条件动态地决定数据的流向
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        ])
        
    def forward(self ,x):
        for layer in self.layers:
            x = layer(x)
        return x

# nn.ModuleList
# 模块按照添加的顺序自动连接，不需要在forward方法中再次指定连接方式，使用起来更加简洁，但灵活性相对较低
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.layers(x)
