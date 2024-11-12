import sys
import package.a
import package.g
print(f"id(sys.modules['package.a']) = {id(sys.modules['package.a'])}")
print(f'in h: x = {package.a.x}')
print(f'in h: id(a) = {id(package.a)}')