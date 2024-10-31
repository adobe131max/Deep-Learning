import sys

from . import b
from ..package2 import c
from .. import d

print('a.py')

for path in sys.path:
    print(path)
