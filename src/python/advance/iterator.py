from collections.abc import Iterable, Iterator

"""
可迭代对象(iterable)和迭代器(iterator)是两个不同的概念
一个对象有__iter__()方法就是可迭代对象, __iter__()返回一个迭代器, 且必须返回迭代器, 也就是说__iter__()方法返回的对象必须有__next__()方法
一个对象有__next__()方法就是迭代器, 调用next()依次返回其中的元素, 到达最后会抛出异常
为什么要区分 iterable 和 iterator? iterable 表明该对象可以迭代, 但实际迭代则交给迭代器完成
"""

a = 1
print(isinstance(a, Iterable))

# list是可迭代的但不是迭代器,list只有__iter__方法没有__next__方法
li = [1, 2, 3]
print(isinstance(li, Iterable))
print(isinstance(li, Iterator))

it = iter(li)
print(isinstance(it, Iterable))
print(isinstance(it, Iterator))

print(next(it))
print(next(it))
print(next(it))

# for遍历可迭代对象时，先通过__iter__获取迭代器，在对迭代器调用__next__方法，并在到达末尾后捕获异常并退出
for i in li:
    print(i)

# 通常会把__iter__方法返回slef，然后给自己添加__next__方法
class Reverse:
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]
    
r = Reverse([1, 2, 3])

print(isinstance(r, Iterable))
print(isinstance(r, Iterator))

for i in r:
    print(i)
