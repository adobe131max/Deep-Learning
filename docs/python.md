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

Python中导入模块时，实际上会把被导入的模块执行一遍

``` py
import module_name                          # 导入模块
import package_name.module_name             # 导入包中的模块
import package_name.module_name as name     # 别名简化

# import 包/模块
# 不可以直接 import module_name.function_name/class_name/variable_name
# 特定对象的导入方式：
from module_name import function_name/class_name/variable_name

# 在包内部的模块中，可以使用相对路径来导入：
from .module1 import function_name
```

import 自己编写的脚本时在当前脚本所在路径下搜索，import 的脚本需要在同一目录或子目录中，否则需要添加搜索路径

自己创建的包需要在目录下创建一个空的 __init__.py 文件将其识别为包

``` py
import sys
print(sys.path)
```

## type

### list

### tuple

### dict
