import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

"""
1. DataLoader如何加载数据
2. 对数据做变换（标签不变）
3. 如何显示原图
4. 如何遍历Dataloader
"""

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # tuple
])

# 加载MNIST数据集
# transform 进行数据增强时，通常不会直接增加数据集的大小。这是因为 transform 的工作方式是在数据加载时实时对每个数据样本进行变换
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
print(f'train dataset len: {len(train_dataset)}')
print(f'test dataset len: {len(test_dataset)}')

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 获取一批训练数据并显示
# iter() 函数用于创建一个迭代器对象，这个对象可以被用来遍历一个序列（如列表、元组、字典等）或其他可迭代对象
dataiter = iter(train_loader)
# next() 用于从迭代器中获取下一个元素。如果迭代器已经遍历完所有元素，则会抛出 StopIteration 异常
images, labels = next(dataiter)
print(images.shape)
print(f'normalized:\n{images[0][0]}')   # 转换后的图片tensor
print(labels)                           # 标签不变

# 反归一化
class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

unnormalize = UnNormalize(mean=(0.1307,), std=(0.3081,))
images = unnormalize(images)
print(f'original:\n{images[0][0]}')     # 转换前的图片tensor

# 转换为可显示的格式并显示
images = images.numpy()
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 2))
for idx, ax in enumerate(axes):
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(f'Label: {labels[idx].item()}')
    ax.axis('off')
plt.show()

print(len(train_loader))            # DataLoader 对象中批次的数量
print(len(train_loader.dataset))    # 数据集的总样本数量。这是数据集对象的长度，不受 DataLoader 的 batch_size 设置影响

# 如何遍历dataloader
# enumerate()将一个可迭代对象（如列表、元组）和一个索引值组合在一起，返回一个迭代器，生成包含索引和值的元组，索引从0开始
i = 0
for batch_idx, (images, labels) in enumerate(train_loader):
    print(f'Batch index: {batch_idx}')
    print(f'Images shape: {images.shape}')  # 应该是 [batch_size, channels, height, width]
    print(f'Labels shape: {labels.shape}')  # 应该是 [batch_size]
    # 这里的 images 和 labels 都是当前批次的数据
    i+=1
    if i > 4:
        break

# 本来就可以遍历
i = 0
for images, labels in train_loader:
    print(f'Images shape: {images.shape}')  # 应该是 [batch_size, channels, height, width]
    print(f'Labels shape: {labels.shape}')  # 应该是 [batch_size]
    # 这里的 images 和 labels 都是当前批次的数据
    i+=1
    if i > 4:
        break
