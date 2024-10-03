import torch
import numpy as np

shape = np.array([0, 1, 2, 3, 4, 5])
src = np.array([[1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9]])

# 负数索引

print(shape[-1]) # 最后一个元素
print(shape[-2]) # 倒数第二个元素

# 切片

# sequence[start:end:step]
# 如果不指定 start 和 end，则默认分别为序列的末尾和开头
# 如果 step 为负数，那么切片将从 start 开始，朝着序列的开头方向，以 step 的绝对值为间隔进行取值，直到到达 end（不包括 end）

print(shape[-2:])
print(shape[:4])    # [0, 4)
print(shape[0:5])
print(shape[0:6])
print(shape[0:7])
print(shape[0:0])
print(shape[0:-1])

# src[:, 0] 选取第0列
column_0 = src[:, 0]
print("src[:, 0]:", column_0)   # 输出 [1 4 7]

# src[:][0] 选取第0行
row_0 = src[:][0]
print("src[:][0]:", row_0)      # 输出 [1 2 3]  src[:][0]实际上是两次切片操作：src[:]：首先，这个操作表示选取所有行，这将返回一个与原始数组相同的数组副本。[0]：接着，从上一步得到的数组中选取第0行。

bbox = torch.tensor([[1, 2, 8, 9]])
print('bbox:', bbox[:, [0, 2]])
bbox[:, [0, 2]] = torch.tensor(10) - bbox[:, [2, 0]]
print('bbox:', bbox)

# 其它操作

# 扩展数组维度
print(shape[:, None, None])     # shape: shape → (*shape, 1, 1) 只能对numpy.ndarray和torch.tensor进行这种操作，不能对list
print(shape[None])              # shape: shape → (1, *shape)
print(src[:, None])
