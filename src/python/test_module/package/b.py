import a

a.x += 1
a.y[0] = 0
# a.y = [0, 1, 2]
print(f'in b: x = {a.x}')
print(f'in b: id(a) = {id(a)}')