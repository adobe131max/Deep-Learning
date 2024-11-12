# $env:PYTHONPATH = "D:\WorkStation\Projects\Deep-Learning\src\python\test_module\package"
import sys
import package.a
import package.b
print(f"id(sys.modules['a']) = {id(sys.modules['a'])}")
print(f"id(sys.modules['package.a']) = {id(sys.modules['package.a'])}")
print(f'in h: x = {package.a.x}')
print(f'in h: id(a) = {id(package.a)}')