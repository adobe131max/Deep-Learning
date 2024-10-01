import time
import torch
import torchvision

# 根据这个测试的结果选择合适的 num_workers 0最快

# 定义数据集转换
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 加载 CIFAR10 数据集
dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

def test_num_workers(num_workers):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=num_workers)
    start_time = time.time()
    for images, labels in data_loader:
        pass
    end_time = time.time()
    return end_time - start_time

if __name__ == '__main__':
    num_workers_values = [0, 1, 2, 4, 8]
    for num_workers in num_workers_values:
        elapsed_time = test_num_workers(num_workers)
        print(f"num_workers={num_workers}, elapsed time: {elapsed_time:.2f} seconds")
