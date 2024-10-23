import time

# decorate函数接受一个函数作为参数，并返回一个新的函数，由新的函数调用被装饰的函数
def decorate(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ans = func(*args, **kwargs)
        time.sleep(1)
        end_time = time.time()
        print(f'{func.__name__} execution time: {end_time - start_time} seconds.')
        return ans
    return wrapper

def add(a, b):
    return a + b

# 使用decorate方法对原方法包装
add = decorate(add)

print(add(1, 2))

# 可以简化成这种写法，效果一样
@decorate
def mul(a, b):
    return a * b

print(mul(2, 3))