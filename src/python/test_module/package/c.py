import a
import b

print(f'in c: x = {a.x}')
print(f'in c: id(a) = {id(a)}')
import sys
print(f"id(sys.modules['a']) = {id(sys.modules['a'])}")