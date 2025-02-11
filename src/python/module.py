import sys

r"""
sys.modules 是一个字典，用于存储已经被导入的模块信息
键是模块的名称（以字符串形式表示），值是对应的已经导入的模块对象本身
"""

print(len(sys.modules))
for module in sys.modules:
    print(module)

# add type to sys.modules
import type

r"""
当使用import语句导入一个模块时,Python 首先会在sys.modules中查找是否已经存在该模块
如果已经存在,就直接返回对应的模块对象
如果不存在,会去查找并执行模块的代码进行导入,并将导入后的模块对象添加到sys.modules字典中
"""

print(len(sys.modules))
for module in sys.modules:
    print(module)

# Cool, right?
sys.modules['type'].test_list()     # equal to type.test_list()

r"""
绝对导入时
import a.b.c 会添加 a、a.b、a.b.c
from a.b import c 也会添加 a、a.b、a.b.c, 当然 c 得是模块而不是变量、函数...
"""

# add src、src.python、src.python.type to sys.modules
import src.python.type  # from src.python import type

print(len(sys.modules))
for module in sys.modules:
    print(module)

# notice that src.python.type and type exist at the same time, so they are two different instances
print(id(sys.modules['type']))
print(id(sys.modules['src.python.type']))

# didn't add copy.deepcopy because it's not a module but a function
from copy import deepcopy

print(len(sys.modules))
for module in sys.modules:
    print(module)
