# Python 模块机制与多模块共享变量

Python 提供了强大的模块系统，使得开发者可以组织代码、共享变量和功能，提高代码的复用性和维护性。在多模块的项目中，理解模块机制、sys.modules 的工作原理以及多模块共享变量的方法，能够帮助开发者更高效地管理代码。本篇文章将深入探讨这些内容，旨在为开发者提供清晰的模块使用和共享变量的最佳实践

## sys.modules

在 Python 中 sys.modules 是一个字典，用于存储已经被导入的模块信息，键是模块的名称（以字符串形式表示），值是对应的已经导入的模块对象本身

当使用 import 和 from ... import ... 导入一个模块时，Python 首先会在 sys.modules 中查找是否已经存在该模块，如果已经存在,就直接返回对应的模块对象，如果不存在，会去查找并执行模块的代码进行导入，并将导入后的模块对象添加到 sys.modules 字典中

``` py
import sys
for module in sys.modules:
    print(module)
```

需要注意的是 sys.modules 中给模块添加的键与导入方式有关，不规范的导入可能会引发一些问题

- `import package1.package2.module1` 会同时添加 package1、package1.package2、package1.package2.module1
- `from package1.package2 import module1` 等价于上面一条，不过使用起来更简单，会直接引入指定的名称到当前命名空间，而不会引入上级包的名称，但如果是 from module import var 则只是在本模块添加 var = moduel.var
- `from . import module` 从顶层 package 开始，添加 top-package、...、top-package...module

## 多模块共享变量

下面结合具体代码解释

structure

``` plain
project/
    package/
        a.py
        b.py
        c.py
        d.py
        e.py
        f.py
        g.py
        i.py
    h.py
    j.py
```

在 a.py 中定义了一些变量：

``` py
# a.py
x = 1
y = [1, 2, 3]
print('a run')
```

### import

在 b.py 和 c.py 中都 import a：

``` py
# b.py
import a

a.x += 1
a.y[0] = 0
print(f'in b: x = {a.x}')
print(f'in b: id(a) = {id(a)}')

# c.py
import a
import b
import sys

print(f'in c: x = {a.x}')
print(f'in c: id(a) = {id(a)}')
print(f"id(sys.modules['a']) = {id(sys.modules['a'])}")
```

`python ./package/c.py` 输出：

``` output
a run
in b: x = 2
in b: id(a) = 1868194330320
in c: x = 2
in c: id(a) = 1868194330320
id(sys.modules['a']) = 1868194330320
```

可见在 b c 模块中使用的 a 和 sys.modules['a'] 是同一个，所以 b c 对 a 的修改互相可见，这也是对需要修改的共享变量使用最简单有效的方式

### from ... import var

``` py
# d.py
from a import x, y

import b
print(f'in d: x = {x}')
print(f'in d: y = {y}')

from a import x, y
print(f'in d: x = {x}')
```

`python ./package/d.py` 输出：

``` output
a run
in b: x = 2
in b: id(a) = 2538161125312
in d: x = 1
in d: y = [0, 2, 3]
in d: x = 2
```

`from a import x, y` 只是在该模块在添加了 `x, y = sys.modules['a'].x, sys.modules['a'].y`，所以 x 的修改不可见，但 y 是可变类型，d和b引用的是同一个对象所以修改可见，但如果是 `a.y = [0, 1, 2]` 修改就不可见

再次 `from a import x, y` 后 x,y 都更新为 a.x,a.y，所以如果是 from import var 需要手动刷新，对需要修改的共享变量不建议使用 from import var

一下代码再次证明了这一点：

``` py
# e.py
from a import x

x += 1
print(f'in e: x = {x}')

# f.py
import a
import e
print(f'in f: x = {a.x}')
```

`python ./package/f.py`，输出：

``` output
a run
in e: x = 2
in f: x = 1
```

### 相对导入

``` py
# g.py
from . import a
a.x += 1
print(f'in g: x = {a.x}')
print(f'in g: id(a) = {id(a)}')

# h.py
import sys
import package.a
import package.g
print(f"id(sys.modules['package.a']) = {id(sys.modules['package.a'])}")
print(f'in h: x = {package.a.x}')
print(f'in h: id(a) = {id(package.a)}')
```

`python h.py`，输出：

``` output
a run
in g: x = 2
in g: id(a) = 1678209465136
id(sys.modules['package.a']) = 1678209465136
in h: x = 2
in h: id(a) = 1678209465136
```

这里使用方式发生了变化，a在 sys.modules 中的 key 从 a 变为 package.a，相对导入和就对导入使用的是同一个模块a，所以修改依然可见

***another example***

``` py
# i.py
from . import a
from . import g

print(f'in j: x = {a.x}')
print(f'in j: id(a) = {id(a)}')
```

`python -m package.i`，这里需要运行含有相对导入的脚本，注意运行方式和之前有所区别。使用的都是 sys.modules['package.a']，所以修改依然可见

``` output
a run
in g: x = 2
in g: id(a) = 2548328315616
in j: x = 2
in j: id(a) = 2548328315616
```

### 不同的 key

接下来复现一些特使情况，不建议实际中这么做

``` py
# j.py
import sys
import package.a
import package.b
print(f"id(sys.modules['a']) = {id(sys.modules['a'])}")
print(f"id(sys.modules['package.a']) = {id(sys.modules['package.a'])}")
print(f'in h: x = {package.a.x}')
print(f'in h: id(a) = {id(package.a)}')
```

`python j.py`，注意这里由于需要 import a，但 sys.path 默认只会添加直接运行的脚本所在的路径，所以需要把 package 所在的路径添加到 PATHONPATH 或 sys.path中，输出：

``` output
a run
a run
in b: x = 2
in b: id(a) = 2074421087104
id(sys.modules['a']) = 2074421087104
id(sys.modules['package.a']) = 2074420002144
in h: x = 1
in h: id(a) = 2074420002144
```

一般一个模块只会运行一次，但这里 a 模块却运行了两次，一次是添加 a，另一次是添加 package.a，且 j 中使用的是 sys.modules['package.a']，b 中使用的是 sys.modules['a']，这和不同模块中 import 模块的方式有关

---

End
