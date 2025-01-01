# Python

## 基础

引用类型：引用类型的变量并不直接存储数据值，而是存储实际数据的地址  
可变数据类型：可以修改对象本身而无需创建一个新的对象
不可变数据类型：无法修改对象本身，只能修改引用的对象（数字、字符串、tuple）

python中变量本身不是独立地占用一个传统意义上的内存地址，更像是一个指针，指向一个内存中的对象
python的赋值运算符=只是建立变量和对象的引用关系而不是复制，这里的引用和C++的引用不是一个东西  
python中一切皆对象，包括数字

## import

### Module & Package

**Module（模块）**：一个py文件就是一个模块，包也是模块
**Package（包）**：一个目录被称为包

Python中导入模块时，实际上会把被导入的模块执行一遍

### 相对导入与绝对导入

<https://zhuanlan.zhihu.com/p/971049360>

``` py
import module_name                          # 导入模块
import package_name.module_name             # 导入包中的模块
import package_name.module_name as name     # 别名简化

# import 包/模块
# import 只能绝对路径导入
# 不可以直接 import module_name.function_name/class_name/variable_name

# from 包/模块 import 包/模块/类/变量
# from import 可以相对导入也可以绝对导入
from module_name import function_name/class_name/variable_name

# 在包内部的模块中，可以使用相对路径来导入(.开头)
from .module1 import function_name
```

1. 绝对导入的搜索路径基于 sys.path
2. 相对导入的搜索路径基于当前包的层级关系，不能相对导入包外的模块，不能直接运行包含相对导入的模块
3. 直接运行脚本 `python module.py`
   1. sys.path 包含直接运行的脚本所在的绝对目录
   2. 不能包含相对导入，因为无法确定包的层级关系
   3. 适合独立脚本的执行
4. 以模块方式运行脚本 `python -m package.module`
   1. sys.path 不包含当前脚本所在的绝对目录，包含工作目录与运行包的绝对目录
   2. 以模块方式运行需要把包添加到 PYTHONPATH 环境变量中 `$env:PYTHONPATH = "D:\project\package"`，或直接在包所在的目录下运行
   3. 可以使用相对导入
   4. 适合包内模块的执行

Tips:

1. 不要直接运行包含相对导入的模块,因为相对导入是基于模块在包结构中的位置进行的，直接运行模块无法确定其在包中的位置,否则会引发`ImportError: attempted relative import with no known parent package`
2. 相对导入只能在同一个顶层package中使用,不能相对导入package外的模块,否则会引发`ValueError: attempted relative import beyond top-level package`

### import 与 from ... import ...

<https://zhuanlan.zhihu.com/p/6380569753>

1. import 和 from import 有什么区别？  
如果 from a import b 的是 module，那么和 import a.b 没什么区别，只是简化了命名空间，等价于 import a.b as b，都会把 a 和 a.b 加入 sys.modules

2. 导入模块和导入变量有什么区别？  
但如果 from import 的是变量/函数...，那么只是一个建立一个引用 `a = sys.modules['moduleA'].a`，如果引用被 = 修改则不可见，如果是修改内部成员则可见(id不变)

3. 是不是一个模块要看导入时添加的 key 是否一样
相对导入 from . import module 会加入 package 和 package.module 到 sys.modules 中（从顶层 package 开始），但如果是 import module 添加的则是 module，此时 module 和 sys.modules 在 sys.modules 中的 key 不一样，所以会加载两次，它们之间的修改互相都不可见

### sys.path

sys.path 是一个列表，包含了 Python 解释器搜索模块时会查找的所有路径。当使用 import 语句进行绝对导入时，Python 会按顺序从这些路径中查找要导入的模块

直接运行脚本时 sys.path 包含直接运行的脚本的路径，以模块方式运行是则会包含当前工作目录

``` py
import sys
print(sys.path)
```

### sys.modules

sys.modules 是一个字典，用于存储已经被导入的模块信息，键是模块的名称（以字符串形式表示），值是对应的已经导入的模块对象本身

当导入一个模块时,Python 首先会在 sys.modules 中查找是否已经存在该模块，如果已经存在,就直接返回对应的模块对象，如果不存在,会去查找并执行模块的代码进行导入，并将导入后的模块对象添加到 sys.modules 字典中

在多个模块导入的同一个模块中，只要每个模块导入的 sys.modules 中的 key 相同，就是共享同一个模块，修改相互可见

## 规范

### 函数注释

``` py
def func(a, b):
   """
   this is a func to add two numbers
   
   Args:
      a(int): num1
      b(int): num2
   Returns:
      int
   Example::
      >>> func(1, 2)
      3
   """
   return a + b
```

## 特性

### list

### tuple

### dict

### set

### 迭代器

### 生成器

### 装饰器

### 上下文管理器

### 元类

### 闭包
