class Human:
    def  __init__(self, name):
        self.name = name
        print('init human')

class Girl(Human):
    def __init__(self, name):
        '''
        __init__构造函数,只能有一个,第一个参数必须是self
        '''
        super().__init__(name)
        print(f"I'm {self.name}")
        
    @staticmethod
    def greet():
        '''
        装饰器定义静态方法
        '''
        print('hi')
        
        
gf = Girl('callan')
print(gf.name)
Girl.greet()
