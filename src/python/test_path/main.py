import sys
# import package1.a
import package.package1.a as a

print('main.py')

for path in sys.path:
    print(path)
    
from package.package1.b import num

print(num)

x = 1

def touch():
    global x, num      # 当在函数内部试图修改一个全局变量时，如果不使用 global 关键字声明，Python 会认为这是在函数内部创建一个新的局部变量，而不是修改全局变量
    num += 1
    x += 1

touch()
print(num)
print(x)

a.add()
print(num)

r'''
运行该脚本
如果你直接运行该脚本,那么pyhton认为该脚本不会在任何顶级package中
从而使得package1,package2作为两个单独的顶级package,而不属于同一个顶级packge
因而package1,package2之间无法相互使用相对引用

1. 直接运行的脚本只使用绝对导入，因为有相对导入的脚本不能直接运行
2. package内的脚本相互调用使用相对导入
3. 直接运行的脚本与package内的脚本分开

https://zhuanlan.zhihu.com/p/971049360
'''