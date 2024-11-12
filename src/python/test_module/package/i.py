import sys
from . import a
from . import g

print(f"id(sys.modules['package.a']) = {id(sys.modules['package.a'])}")
print(f'in j: x = {a.x}')
print(f'in j: id(a) = {id(a)}')