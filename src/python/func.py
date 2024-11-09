
r"""
函数的默认参数只会创建一次
如果直接赋值默认参数,实际上是创建了一个新的局部变量,不会改变下次调用的默认参数
如果对默认参数内部进行修改会直接修改这个默认对象本身
如果不希望出现这种情况,建议在函数内部使用None作为默认值
"""

class Info():
    def __init__(self, num):
        self.num = num

def test(num = 1, info1 = Info(1), info2 = Info(1)):
    print(num)
    num = 2
    print(info1.num)
    info1 = Info(2)     # this will not change default param, only create a new local variable
    print(info2.num)
    info2.num = 2       # change default param
    
test()
test()
