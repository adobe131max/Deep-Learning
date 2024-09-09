import time
import datetime

print(10 // 3)  # 取整
print(10 / 3)   # 保留小数
print(10 % 3)   # 取模

print([i for i in range(10)])       # [0, 10)
print([i for i in range(1, 10)])    # [1, 10)

print(time.time())
print(datetime.timedelta(seconds=int(time.time())))
print(time.time() / 60 / 60 / 24)

a = [1, 2]
b = [3, 4, 5, 6]
c = a + b           # 拼接
print(c)

# 当一个对象实现了__iter__方法时，它可以被用于for循环和其他迭代上下文
# 当一个对象实现了__getitem__方法时，可以使用方括号[]来获取特定索引位置的元素

batch = [[0,1],[2,3],[4,5],[6,7],[8,9]]
print(*batch)
print(zip(*batch))  # zip 将多个可迭代对象中的元素按照索引位置一一对应地组合起来，生成一个新的可迭代对象
print(list(zip(*batch)))
inp, out = list(zip(*batch))
print(inp, out)

t = (1,2,3)
print(list(t))      # 将一个可迭代对象（如字符串、元组、集合、生成器等）换为一个列表

# inf 代表无穷大 infinity
a = float('inf')
b = float('-inf')
print(a > 1000000)  # True
print(b < -1000000) # True
