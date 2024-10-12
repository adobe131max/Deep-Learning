import sys

# 第一个路径为当前脚本所在的路径
for path in sys.path:
    print(path)

# 绝对路径导入
# 包内模块不推荐使用绝对导入,在被调用时会出问题
import grammar

# 相对路径导入 ❌ 不能直接运行包含相对导入的模块,只能被调用
from . import grammar

grammar.func(age=20)

# 详见 ./testpath
