# import package1.a
import package.package1.a

print('main.py')

r'''
运行该脚本
如果你直接运行该脚本,那么pyhton认为该脚本不会在任何顶级package中
从而使得package1,package2作为两个单独的顶级package,而不属于同一个顶级packge
因而package1,package2之间无法相互使用相对引用
'''