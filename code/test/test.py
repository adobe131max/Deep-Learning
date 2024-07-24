import torch
import math
import numpy as np

print(torch.randint(0, 100, (10, 2)))
print(math.sqrt(256))

src = np.array([[1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9]])

# src[:, 0] 选取第0列
column_0 = src[:, 0]
print("src[:, 0]:", column_0)  # 输出 [1 4 7]

# src[:][0] 选取第0行
row_0 = src[:][0]
print("src[:][0]:", row_0)  # 输出 [1 2 3]  src[:][0]实际上是两次切片操作：src[:]：首先，这个操作表示选取所有行，这将返回一个与原始数组相同的数组副本。[0]：接着，从上一步得到的数组中选取第0行。
