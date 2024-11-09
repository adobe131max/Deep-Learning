import sys

for path in sys.path:
    print(path)
    
print(__package__)

from . import b
from ..package2 import c
from .. import d

print('a.py')
