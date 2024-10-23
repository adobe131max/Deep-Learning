from contextlib import contextmanager

"""
with + 上下文管理器（定义进入和退出 with 的行为）
即使异常退出也会执行退出行为
"""

with open('TODO.md') as f:
    pass

# 通过类实现上下文管理器
# __enter__定义进入的行为，返回值绑定到 as 的变量，__exit__定义退出的行为
class MyOpen:
    def __init__(self, filepath):
        self.filepath = filepath
        
    def __enter__(self):
        print('enter with.')
        return self.filepath

    def __exit__(self, exc_type, exc_value, trace_back):
        print('exit with.')

with MyOpen('TODO.md') as f:
    # raise ValueError('a ValueError occured.')
    print(f'in with: {f}')

# 通过生成器函数实现上下文管理器
# yield 之前的代码在进入时执行
# yield 返回的值绑定到 as 的变量
# yield 之后的代码在退出时执行
# 但生成器函数实现的上下文管理器在with内出现异常时不会自动执行yield后的内容，需要手动捕获yield异常，在finally中执行退出行为
# 需要使用 @contextmanager 装饰生成器函数
@contextmanager
def my_open(filepath):
    print('enter with.')
    try:
        yield filepath
    finally:
        print('exit with.')

with my_open('TODO.md') as f:
    # raise ValueError('a ValueError occured.')
    print(f'in with: {f}')
