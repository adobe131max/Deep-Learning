# * ** 解包
layers = [1, 2, 3]
print(layers)
print(*layers)
print([*layers])

first, *rest = layers
print(first)
print(rest)

def func(**kwargs):
    for key, value in kwargs.items():
        print(f'{key} = {value}')

func(a=1, b=2, c=3)

dict1 = {'a': 1,'b': 2}
dict2 = {'c': 3,'d': 4}
merged = {**dict1,**dict2}
print(merged)

batch = [[0,1],[2,3],[4,5],[6,7],[8,9]]
print(*batch)
print(zip(*batch))          # zip 将多个可迭代对象中的元素按照索引位置一一对应地组合起来，生成一个新的可迭代对象
print(list(zip(*batch)))
inp, out = list(zip(*batch))
print(inp, out)
