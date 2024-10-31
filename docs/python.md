# Python

## 基础

引用类型：引用类型的变量并不直接存储数据值，而是存储实际数据的地址  
可变数据类型：修改变量时，变量的id不变  
不可变数据类型：修改变量时，不会修改变量引用的值，而是建立新的引用，变量的id改变（数字、字符串、tuple）

python中变量本身不是独立地占用一个传统意义上的内存地址，更像是一个标签，指向一个内存中的对象
python的赋值运算符=只是建立引用而不是复制  
python中一切皆对象，包括数字

## Module & Package

**Module（模块）**：一个py文件就是一个模块  
**Package（包）**：一个包含 “init.py” 文件的目录被称为包

### import

<https://zhuanlan.zhihu.com/p/971049360>

Python中导入模块时，实际上会把被导入的模块执行一遍

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
# 绝对路径导入包内部模块在被调用会报错
from .module1 import function_name
```

### sys.path

sys.path 是一个列表，包含了 Python 解释器搜索模块时会查找的所有路径。当使用 import 语句进行绝对导入时，Python 会按顺序从这些路径中查找要导入的模块。

``` py
import sys
print(sys.path)
```

### 总结

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
