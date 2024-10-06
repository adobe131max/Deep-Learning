# python中所有数据类型几乎都是对象
# 在 Python 中进行赋值操作时，实际上是创建了一个对象的引用

a = [1, 2, 3]
b = a

print(id(a))
print(id(b))

c = 1
d = c

print(id(c))
print(id(d))

c = 0
print(id(c))
print(id(d))
print(c)
print(d)

e = 1
print(id(e))
print(id(1))
e = e + 1
print(id(e))

# 函数参数传递也是传递了对象的引用

def modify_list(lst):
    lst.append(4)

my_list = [1, 2, 3]
modify_list(my_list)
print(my_list)  # [1, 2, 3, 4]

# 如果在函数内部尝试重新赋值参数指向一个新的对象，这不会影响到外部的变量

def reassign_list(lst):
    lst = [4, 5, 6]     # 已经修改了指向的对象
    lst.append(7)

my_list = [1, 2, 3]
reassign_list(my_list)
print(my_list)  # [1, 2, 3]（没有被修改）

my_list = []
a = [1, 2, 3]
my_list.append(a)
my_list.append(a)
a[0] = 0
print(my_list)
