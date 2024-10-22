import numpy as np

"""
[start : stop : step]
开始索引（含）默认为 0
结束索引（不含）默认为 len()
间隔，默认为 1
索引也可以是负数，索引为-n表示倒数第n个（-1为最后一个，0只代表第一个，不能表示最后一个的后一个）
间隔为负数表述从后往前
"""

arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 传统切片
print(arr[:3])
print(arr[7:])
print(arr[6:9])
print(arr[::2])
print(arr[-1])
print(arr[-2:])
print(arr[-3:-1])
print(arr[-3:0:-1])

print('\n<--- slice --->\n')

# 使用slice
sli1 = slice(5)                 # stop
sli2 = slice(2, 4)              # start, stop
sli3 = slice(0, 10, 3)          # start , stop, step
sli4 = slice(5, None)           # = [5:] 使用默认值也需要传 None
sli5 = slice(None, None, 3)     # = [::3]

print(arr[sli1])
print(arr[sli2])
print(arr[sli3])
print(arr[sli4])
print(arr[sli5])

print('\n<--- slice for ndarray and tensor --->\n')

# 只有numpy.ndarray torch.tensor 支持的高级切片，list、tuple不支持
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# 这并不是选择第一列 arr[:] = arr，还是选的第一行
print(arr[:][0])
print(f'column 0: {arr[:, 0]}')     # 连续选择
print(arr[[0, 2]])                  # 选择多个指定元素
print(arr[[0, 0, 2], [0, 2, 1]])    # = [ arr[0][0], arr[0][2], arr[2][1] ]

# 扩展数组维度
print(arr[:, :, None])  # shape: shape → (3, 3, 1) 在最后增加一个维度
print(arr[:, None])     # shape: shape → (3, 1, 3) 在第一维后面增加一个维度
print(arr[None])        # shape: shape → (1, 3, 3) 在最前面增加一个维度