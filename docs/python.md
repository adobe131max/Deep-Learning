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

## type

### list

### tuple

### dict
