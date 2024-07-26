import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)     # 通道数由1变为6，输出的通道数 = 卷积核的个数，feature map
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)      # 展平, -1 自动推导为 batch size
        x = F.relu(self.fc1(x))     # 如果你认真看了，可能会有疑惑：fc1不是一个对象吗？fc1()调用的是__call__方法，实际上就是forward方法
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),                      # 将图像转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化为均值0.1307，标准差0.3081
])

# ./data 表示在当前工作目录下的 data 文件夹。
# train=True/train=False：指定是训练集还是测试集
# download=True：如果数据集不存在则下载
# transform=transform：应用前面定义的转换
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)   # batch_size 批量训练，一般越大越好，需要的内存多
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)  # shuffle=True/shuffle=False：是否在每个epoch开始时打乱数据。训练数据一般需要打乱，测试数据则不需要

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train(epoch):
    model.train()   # 进入训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # PyTorch中的梯度是累加的（默认情况下，梯度在每次 backward() 时都会累加到现有的梯度上），所以在每次训练迭代开始时，需要将上一次迭代的梯度清零，如果不清除会导致梯度爆炸和梯度更新错误
        output = model(data)                # 前向传播，计算输出
        loss = criterion(output, target)    # 计算损失
        loss.backward()                     # 反向传播，计算每个参数的梯度
        optimizer.step()                    # 更新模型参数
        if batch_idx % 100 == 0:            # 100个batch输出一次loss，也就是6400条数据
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 测试模型
def test():
    model.eval()    # 进入评估模式 Dropout 层会停止随机丢弃神经元。BatchNorm 层会使用训练期间计算的均值和方差，而不是批量数据的均值和方差。
    test_loss = 0
    correct = 0
    with torch.no_grad():   # 评估阶段不计算梯度，不创建计算图
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

print(len(train_loader))
print(len(train_loader.dataset))

# 执行训练和测试
for epoch in range(1, 5):
    train(epoch)
    test()
