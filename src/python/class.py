"""
类默认继承 Object
__init__是构造函数, 只能定义一个
成员使用 self. 定义在构造函数中
成员方法定义时的第一个参数必须是self, 调用时会自动传入
类变量（静态变量）定义在构造函数外，不带 self
类方法使用装饰器 @classmethod, 第一个参数为类本身, 调用时会自动传入
静态方法使用装饰器 @staticmethod, 不需要传入类
"""

class Student:
    num = 0
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        Student.num += 1
    
    @classmethod
    def from_string(cls, info):
        name, age = info.split(' ')
        age = int(age)
        return cls(name, age)
    
    @staticmethod
    def func():
        print('static method called.')


s1 = Student('abc', 18)
print(Student.num)
s2 = Student.from_string('elc 21')
print(Student.num)


class Human:
    def  __init__(self, name):
        self.name = name
        print('init human')

class Girl(Human):
    def __init__(self, name):
        super().__init__(name)
        print(f"I'm {self.name}")
        
    @staticmethod
    def greet():
        print('hi')
        
        
gf = Girl('callan')
print(gf.name)
Girl.greet()
