import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc1(x)
        x = F.relu(x)       # 不在模型的结构中，children() 和 modules() 中都不会包含
        return x


def save_and_load(path = './model.pth'):
    # 1. 保存完整模型（包括结构与权重）
    torch.save(model, path)
    # 完整模型加载
    model = torch.load(path)

    # 2. 只保存模型权重
    torch.save(model.state_dict(), path)
    # 需要先创建模型
    model = resnet50()
    # 载入模型权重
    # strict=False
    # 如果加载的权重文件中的键与模型中的参数名称不完全匹配，或者模型中有一些参数在权重文件中不存在，不会引发错误。
    # 对于模型中存在但权重文件中不存在的参数，这些参数将保持其初始状态
    model.load_state_dict(torch.load(path))


def model_structure():
    model = Model()

    # 模型的下一层子结构
    # 只包含通过 self.xxx = ... 形式注册的 nn.Module 子类
    print('model.named_children():')
    for name, module in model.named_children():
        print(name)

    # 模型的所有模块（递归遍历所有模块）
    # 第一个模块是模型本身，name 为空
    print('model.named_modules():')
    for name, module in model.named_modules():
        print(name)

    # 模型的所有参数
    print('model.named_parameters():')
    for name, param in model.named_parameters():
        print(name)


# nn.ModuleList
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


# nn.Suquential
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
    

if __name__ == '__main__':
    model_structure()
