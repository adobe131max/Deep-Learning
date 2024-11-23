import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 二分类
# BCEWithLogitsLoss 会对预测结果进行 sigmoid 处理而 BCELoss 则不会
# BCEWithLogitsLoss 等价于 F.binary_cross_entropy_with_logits
criterion = nn.BCEWithLogitsLoss()
predict = torch.tensor([10, 4, -5], dtype=torch.float32)
target = torch.tensor([1, 1, 0], dtype=torch.float32)       # target 需要是 0 或 1 的浮点数
loss = criterion(predict, target)
print(loss)

# 验证
predict = F.sigmoid(predict)
print(predict)
loss0 = -math.log(predict[0])
loss1 = -math.log(predict[1])
loss2 = -math.log(1 - predict[2])
print((loss0 + loss1 + loss2) / 3)

# 多分类
# 定义 loss 为 Cross Entropy Loss
# 等价于 F.cross_entropy()
criterion = nn.CrossEntropyLoss()                  

# 预测概率（这里并不需要进行softmax，CrossEntropyLoss中已经包含了softmax）
predict = torch.tensor([[1.0, 9.0, 21.0], [3.0, 13.0, 7.0]])    # 注意类型是float

# 真实类别
target0 = torch.tensor([0, 0])
target1 = torch.tensor([1, 1])
target2 = torch.tensor([2, 2])

# 计算平均 loss
loss0 = criterion(predict, target0)
loss1 = criterion(predict, target1)
loss2 = criterion(predict, target2)
print(loss0)
print(loss1)
print(loss2)

# 通过手工计算来证明不需要 softmax
predict = torch.tensor([[0.1, 0.2, 0.7]])
target0 = torch.tensor([0])
target1 = torch.tensor([1])
target2 = torch.tensor([2])

loss0 = criterion(predict, target0)
loss1 = criterion(predict, target1)
loss2 = criterion(predict, target2)
print(loss0)
print(loss1)
print(loss2)

sum = math.exp(0.1) + math.exp(0.2) + math.exp(0.7)
print(math.log(math.e))
print(-math.log(math.exp(0.1) / sum))
print(-math.log(math.exp(0.2) / sum))
print(-math.log(math.exp(0.7) / sum))
