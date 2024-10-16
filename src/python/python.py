import copy
import torch

# python中一切皆对象，赋值运算符=只是建立引用而不是复制 
# 每个变量可以理解为只是其引用的地址
a = 1           # a储存的是1的地址
b = a           # b也储存的是1的地址
print(id(a))    # id是变量中存储的地址
print(id(b))
a = 2           # python中数字是不可变类型，a储存的是2的地址
print(b)

c = [1, 2, 3]
d = c           # c，d有相同的引用
print(id(c))
print(id(d))
c[0] = 0
print(d)

a = [1, 2, 3]
b = a
a = [4, 5, 6]
print(b)        # a、b虽然有相同的引用，但a、b本身是独立的

a = torch.Tensor([1, 2, 3])
b = a
b /= 2          # 原地操作，直接修改了a，共享内存慎用
print(a)
print(id(a))
print(id(b))
a = torch.Tensor([1, 2, 3])
b = a
b = b / 2       # 创建新的tensor
print(a)
print(id(a))
print(id(b))

print('\n<--- copy --->\n')

# 浅拷贝
a = [1, 2, [3, 4]]
b = copy.copy(a)        # 只拷贝一层
print(id(a))
print(id(b))
a[0] = 0
print(b)
print(id(a[2]))
print(id(b[2]))
a[2][0] = 0
print(b)

# 深拷贝
a = [1, 2, [3, 4]]
b = copy.deepcopy(a)    # 只要是可变类型就继续深拷贝
print(id(a[2]))
print(id(b[2]))
a[2][0] = 0
print(b)

# 所有的变量都相当于C++的指针
# 直接赋值形参不会改变实参
def assign(x):
    x = [1, 1, 1]

def change(x):
    x[0] = 0
    
x = [1, 2, 3]
assign(x)
print(x)
change(x)
print(x)
