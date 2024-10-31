import sys
# import package1.a
import package.package1.a

print('main.py')

for path in sys.path:
    print(path)

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