# Python

## import

Python中导入模块时，实际上会把被导入的模块执行一遍

1. 导入模块

``` py
import torch            # torch是模块
import torch.nn         # torch.nn也是模块
import torch.nn as nn   # 简化，nn也是模块
```

2. 直接导入模块/函数/变量/类，简化命名空间

torch.utils.data是模块  
torch.utils.data.DataLoader是类

``` py
from torch.utils.data import DataLoader
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
