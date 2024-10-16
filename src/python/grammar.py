from typing import List

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

age: int = 25
name: str = "Alice"
is_active: bool = True

def add_numbers(a: int, b: int) -> int:
    return a + b

numbers: List[int] = [1, 2, 3]
