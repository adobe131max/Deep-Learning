import os
import sys

# 获取当前工作目录，也就是命令行里执行的该脚本命令前面的路径
print(f'\npwd: {os.getcwd()}')

# PYTHONPATH 环境变量的内容会被添加到 sys.path
print(f'\nPYTHONPATH: {os.environ.get("PYTHONPATH", None)}')

r"""
sys.path 包含了 Python 解释器搜索模块时会查找的所有路径，使用绝对导入时会从这些路径中查找

1. 直接运行脚本时, sys.path 包含当前脚本所在的绝对目录
    `python ./src/python/sys_path.py`
2. 以模块方式运行脚本时, sys.path 不包含当前脚本所在的绝对目录，但包含工作目录和运行包的绝对目录
    以模块方式运行需要把包添加到 PYTHONPATH 环境变量中，或直接在包所在的目录下运行
    通过 `dir env: ` 查看所有环境变量
    `$env:PYTHONPATH = "D:\WorkStation\Projects\Deep-Learning\src"` 添加该包的绝对路径到环境变量中
    `python -m src.python.sys_path`
"""
print(f'\nsys.path:')
for path in sys.path:
    print(path)

print(f'\n__name__: {__name__}')
print(f'\n__package__: {__package__}')
