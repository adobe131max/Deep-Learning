# x = a.x, y = a.y
from a import x, y

import b
print(f'in d: x = {x}')
print(f'in d: y = {y}')

from a import x, y
print(f'in d: x = {x}')