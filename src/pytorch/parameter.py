import torch

from torchvision.models import resnet50

model = resnet50()

for param in model.parameters():
    print(param.requires_grad)

model.requires_grad_(False)

for param in model.parameters():
    print(param.requires_grad)

for name, param in model.named_parameters():
    print(name)
