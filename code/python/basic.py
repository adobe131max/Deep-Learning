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
c = a + b
print(c)
