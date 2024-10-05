import copy

# python中一切皆对象，赋值运算符=只是建立引用而不是复制 
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
