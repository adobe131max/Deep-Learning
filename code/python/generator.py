
'''
generator 生成器是通过使用生成器函数或者生成器表达式创建的。生成器函数中包含至少一个yield语句
当调用生成器函数时，它不会立即执行函数体，而是返回一个生成器对象
这个生成器对象可以通过next()方法或者在循环中进行迭代来逐步执行生成器函数中的代码
每次执行到yield语句时,生成器函数会暂停并返回一个值,下次调用next()方法时，它会从上次暂停的地方继续执行。
'''

def my_generator():
    yield 1
    yield 2
    yield 3

# 使用生成器函数
gen = my_generator()
print(type(gen))
print(next(gen))
print(next(gen))
print(next(gen))
