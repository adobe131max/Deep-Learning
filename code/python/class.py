class Girl:
    def __init__(self, name):
        '''
        __init__构造函数,只能有一个,第一个参数必须是self
        '''
        self.name = name
        print(name)
        print(self.name)
        
    @staticmethod
    def fuck():
        '''
        装饰器定义静态方法
        '''
        print('fuck me')
        
        
gf = Girl('福利姬')
print(gf.name)
Girl.fuck()
