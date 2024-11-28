import torch

from torchvision.models import resnet18

model = resnet18()

for param in model.parameters():
    print(param.requires_grad)

# 冻结部分模块参数
model.layer1.requires_grad_(False)

print('<------>')
for param in model.parameters():
    print(param.requires_grad)

# 一个完整的模块，比如 conv、bn
for name, module in model.named_modules():
    print(name)

# 参数 weights、bias
for name, param in model.named_parameters():
    # layer1.0.conv1.weight
    print(name)
    
for name, param in model.layer1.named_parameters():
    # 0.conv1.weight
    print(name)
