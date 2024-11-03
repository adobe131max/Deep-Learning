import time

# 装饰器函数实现
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

# 如果装饰器需要参数就在外面再套一层函数
def param_decorator(t=1):
    def decorater(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            ans = func(*args, **kwargs)
            time.sleep(t)
            end_time = time.time()
            print(f'{func.__name__} execution time: {end_time - start_time} seconds.')
            return ans
        return wrapper
    return decorater

@param_decorator(0.5)
def say_hi():
    print('hi')
    
say_hi()

# TODO：装饰器类实现
# 实现一个类，类中实现__call__方法，在__call__方法中实现装饰逻辑

# torch.no_grad 实际上是通过 contextlib.ContextDecorator 实现的
# 可以同时将一个上下文管理器用作装饰器，而不需要额外实现 __call__
