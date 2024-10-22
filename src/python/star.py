# * ** 解包

# * 解包序列
layers = [1, 2, 3]
print(layers)
print(*layers)
print([*layers])

# ** 解包字典
def print_info(name, age):
    print(f'name: {name}')
    print(f'age: {age}')

info = {
    'name': 'sam',
    'age': 24
}

print_info(**info)

# * ** 打包

# *args 将多个变量打包 args为tuple, 但rest为list
first, *rest = (1, 2, 3)
print(first)
print(type(rest))
print(rest)

def example(*args):
    print(type(args))
    print(args)
    
example(1, 2, 3)

# * 合并序列
a = [1, 2, 3]
b = (4, 5, 6)
c = [*a, *b]
print(c)

# **kwargs 将多个参数打包成一个dict: kwargs
def func(**kwargs):
    print(kwargs)
    for key, value in kwargs.items():
        print(f'{key} = {value}')

func(a=1, b=2, c=3)

# ** 合并字典
dict1 = {'a': 1,'b': 2}
dict2 = {'c': 3,'d': 4}
merged = {**dict1,**dict2}
print(merged)
