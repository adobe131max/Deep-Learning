import torch
import numpy

# TODO
# clamp

def attribute():
    '''
    tensor的属性
    '''
    pass

def test_dtype():
    '''
    torch.Tensor中的dtype元素类型
    '''
    print(torch.int16 == torch.short)
    print(torch.int32 == torch.int)
    print(torch.int64 == torch.long)
    print(torch.float16 == torch.half)
    print(torch.float32 == torch.float)
    print(torch.float64 == torch.double) 

def change_shape():
    '''
    改变tensor的形状:
    reshape     如果内存中连续和view一样,如果不连续则创建一个新的tensor
    view        要求在内存中连续，返回新的视图，共享底层数据
    flatten     展平
    squeeze     去除维度为1的维度
    unsqueeze   增加维度
    unbind      拆分
    '''
    print('\n--- view & reshape ---\n')
    x = torch.tensor([[1,2,3],[4,5,6]])
    print(x.view(3, 2))
    x = x.permute(1, 0)
    print(x)
    # print(x.view(-1))     # 不能view
    print(x.reshape(-1))    # -1 表示自动推导
    
    print('\n--- flatten ---\n')
    x = torch.tensor([[[1],[2],[3]],[[4],[5],[6]]])
    print(x.shape)
    print(x.flatten())      # 默认展平成一维
    print(x.flatten(0, -2)) # -2表示倒数第二维，将[0, -2]维展平为一维
    
    print('\n--- squeeze ---\n')
    print(torch.tensor([[1],[2],[3]]).squeeze())
    
    print('\n--- unsqueeze ---\n')
    x = torch.tensor([1, 2, 3])
    print(x.unsqueeze(0))
    print(x.unsqueeze(1))
    
    print('\n--- unbind ---\n')
    bboxs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    xmin, ymin, xmax, ymax = bboxs.unbind(1)
    print(xmin)
    print(torch.stack((xmin, ymin, xmax, ymax), dim=1))

def test_create():
    '''
    创建tensor
    torch.Tensor()
    torch.tensor()    复制数据
    torch.as_tensor() 不复制数据，共享内存
    '''
    a = numpy.array([1, 2, 3])      # numpy默认int32 torch默认int64
    
    x = torch.tensor(5)
    print(x.dim())          # 0维tensor

    x = torch.as_tensor(a)  # dtype与传入的dtype相同
    print(x)
    print(x.shape)
    print(x.dtype)
    print(a.dtype)

    x = torch.tensor([[1,2,3],[4,5,6]]) # 默认int64
    print(x)
    print(x.shape)
    print(x.dtype)
    print(x.dim())

    x = torch.Tensor([[1,2,3],[4,5,6]]) # 默认float32 torch.Tensor() == torch.FloatTensor()
    print(x)
    print(type(x))                      # torch.Tensor
    print(x.shape)
    print(x.size())                     # .shape = .size()
    print(x.dtype)                      # tensor中元素的类型

    print(torch.Tensor(2,3))            # 指定shape创建tensor，内容未初始化
    print(torch.IntTensor(2,3))         # 指定类型的tensor
    print(torch.randint(0, 10, (3, 4))) # [low high) shape
    print(torch.zeros(2,3))             # 全 0
    print(torch.ones(2,3))              # 全 1
    print(torch.randn(2,3))             # 浮点型随机数 - 均值为0，方差为1，服从b标准正态分布
    print(torch.rand(2,3))              # 浮点型随机数 - [0, 1) 均匀分布
    print(torch.arange(1,10))           # [start, end)

def compute():
    '''
    tensor计算
    '''
    print(f'add:\n{x.add(1)}')          # 生成一个新的tensor
    print(x)
    print(f'add_:\n{x.add_(1)}')        # 带_的直接在原tensor上改变
    print(x)
    print(f'zero_:\n{x.zero_()}')       # 置 0

    print('\n<--- operate --->\n')

    x = torch.Tensor([[1,2,3],[4,5,6]])
    y = torch.tensor([[-1,-2,-3],[-4,-5,-6]])
    print(x+y)
    print(x)
    x+=y        # 直接改变原tensor
    print(x)

    x = torch.tensor([1])
    print(x.item())     # tensor to num
    
def process():
    '''
    torch.max       返回两个tensor: 最大值和索引
    torch.nonzero   返回非0的索引
    torch.eq        返回一个相同shape的tensor,每个元素为 True or False
    torch.where     返回dimension个tensor,每个tensor的元素表示在该维度上的索引
    '''
    print('\n--- max ---\n')
    x = torch.tensor([[1,2,3,4],
                      [4,5,6,7],
                      [7,8,9,0]])
    val, idx = x.max(dim=0)     # dim=0 是寻找每列的最大元素
    print(val)
    print(idx)
    val, idx = x.max(dim=1)     # dim=0 是寻找每行的最大元素
    print(val)
    print(idx)
    
    print('\n--- nonzero ---\n')
    x = torch.tensor([0, 1, -1, 0, 9, 4])
    print(x.nonzero())
    
    print('\n--- eq ---\n')
    print(x.eq(0))
    y = torch.tensor([[0, 1, 2],
                      [0, 1, 0]])
    print(y.eq(0))
    
    print('\n--- where ---\n')
    print(torch.where(x.eq(0)))
    print(torch.where(y.eq(0)))
    

def tensors():
    '''
    多个tensor之间的操作
    stack cat 返回新的tensor,不共享数据
    '''
    print('\n<--- stack --->\n')
    
    # stack会在指定的dimension前插入一个长度为输入tensor个数的dimension
    x = torch.Tensor([[1,2,3],[4,5,6]])
    y = torch.tensor([[-1,-2,-3],[-4,-5,-6]])
    print(torch.stack((x,y)))
    print(torch.stack((x,y)).shape)
    print(torch.stack((x,y),0))
    print(torch.stack((x,y),0).shape)
    print(torch.stack((x,y),1))
    print(torch.stack((x,y),1).shape)
    print(torch.stack((x,y),2))
    print(torch.stack((x,y),2).shape)

    print('\n<--- cat --->\n')

    # cat的dimension的shape会相加，但其余dimension的shape必须相同
    x = torch.Tensor([[1,2,3],[4,5,6]])
    y = torch.tensor([[-1,-2,-3],[-4,-5,-6]])
    xy = torch.cat((x, y))
    print(xy)
    print(xy.shape)
    y = torch.tensor([[0, -1,-2,-3],[0, -4,-5,-6]])
    xy = torch.cat((x, y), dim=1)
    print(xy)
    print(xy.shape)

if __name__ == '__main__':
    # change_shape()
    process()
