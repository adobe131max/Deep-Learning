import torch.utils.data as data

# 自定义数据集

class MyDataset(data.Dataset):
    def __init__(self):
        pass
    
    def __getitem__(self, index):
        pass
    
    # 用于内置函数 len()
    def __len__(self):
        pass
