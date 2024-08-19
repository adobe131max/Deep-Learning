import os
import sys

# Python 解释器在导入模块时搜索模块的路径
print(sys.path)

# 获取当前工作目录 cwd(current working directory)，也就是命令行里执行的该脚本命令前面的路径
# PS D:\WorkStation\Projects\Deep-Learning> & D:/Programs/anaconda3/envs/pytorch/python.exe "d:/WorkStation/Projects/Deep-Learning/code/python/operating system.py"
# 上面的工作目录就是》前的 D:\WorkStation\Projects\Deep-Learning
cwd = os.getcwd()
print(f'当前工作目录: {cwd}')

# os.path.join 用于智能地将一个或多个路径组件连接起来。根据不同操作系统的路径分隔符来处理路径
print(os.path.join(cwd, r'a\b\c'))
file_path = os.path.join(cwd, './code/python/path.py')
print(file_path)
assert os.path.exists(file_path), './code/python/path.py not found.'    # 可以识别/和\

# 获取脚本文件的完整路径
file_path = os.path.abspath(__file__)
print(f'此脚本文件路径: {file_path}')

# 获取脚本文件所在的目录
directory_path = os.path.dirname(file_path)
print(f"此脚本文件所在目录：{directory_path}")

data_path = os.path.join(cwd, "data")
print(data_path)
assert os.path.exists(data_path), f'path {data_path} doesn\'t exist.'

# 路径下所有文件及文件夹名
print(os.listdir(cwd))

# 路径都是相对于工作目录
assert os.path.exists('./README.md'), './README.md not found.'
assert os.path.exists('./code/python/path.py'), './code/python/path.py not found.'
