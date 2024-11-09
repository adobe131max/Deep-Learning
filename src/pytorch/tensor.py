import torch
import numpy

# TODO
# clamp

def attribute():
    '''
    tensor的属性
    '''
    x = torch.randn((2, 3))
    print(x.dim())
    print(x.shape)
    print(x.dtype)
    print(x.device)
    print(x.requires_grad)

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
    
    x = torch.tensor([1, 2, 3])
    y = x.float()   # to float32
    z = y.int()     # to int32
    print(x.dtype)
    print(y.dtype)
    print(z.dtype)
    
    x = torch.tensor(1)
    print(x)
    a = x.item()    # 对于单个数值的张量，返回一个标准的 Python 数值类型
    print(a)

def change_shape():
    '''
    改变tensor的形状:
    reshape     如果内存中连续和view一样,如果不连续则创建一个新的tensor
    view        要求在内存中连续，返回新的视图，共享底层数据
    flatten     展平
    squeeze     去除维度为1的维度
    unsqueeze   增加维度
    permute     重新排列
    transpose   转置
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
    
    print('\n--- permute ---\n')
    x = torch.tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8]])
    y = x.permute(1, 0)         # 交换0、1维上的shape，效果类似 transpose
    print(y)
    print(y.is_contiguous())    # 和 view、reshape不同，permute 改变了元素的底层排列顺序（展平成一维看）
    
    print('\n--- transpose ---\n')
    z = x.transpose(0, 1)       # 和 permute 一样，但一次只能操作两个维度
    print(z)
    print(z.is_contiguous())
    
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
    print(type(x.shape))    # torch.Size
    print(type(x.shape[0])) # int
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
    x = torch.randn(2,3)                # 浮点型随机数 - 均值为0，方差为1，服从b标准正态分布
    print(x.dtype)                      # float32
    print(torch.rand(2,3))              # 浮点型随机数 - [0, 1) 均匀分布
    print(torch.arange(1,10))           # [start, end)
    print(torch.rand(1))


def base_create():
    """
    基于已有tensor创建新tensor
    torch.repeat_interleave         重复张量中的元素
    """
    print('\n--- repeat interleave ---\n')
    x = torch.tensor([
        [[1, 2, 3]],
        [[4, 5, 6]]
    ])
    y = x.repeat_interleave(2, dim=1)   # 指定dim的shape会乘以指定的次数
    print(y)
    print(x.shape)
    print(y.shape)


def compute():
    '''
    tensor计算
    '''
    x = torch.tensor([1, 2, 3])
    print(f'add:\n{x.add(1)}')          # 生成一个新的tensor
    print(x)
    print(f'add_:\n{x.add_(1)}')        # 带_的直接在原tensor上改变
    print(x)
    print(f'zero_:\n{x.zero_()}')       # 置 0

    print('\n<--- operate --->\n')

    x = torch.Tensor([[1,2,3],[4,5,6]])
    y = torch.tensor([[-1,-2,-3],[-4,-5,-6]])
    print(x+y)
    z = x
    x += y          # 直接改变原tensor，如果不想影响z，应该使用 x = x + y，会创建一个新的tensor
    print(x)
    print(z)
    print(id(x) == id(z))
    
    a = torch.tensor(2)
    b = torch.tensor(3)
    print(a * b)

    x = torch.tensor([1])
    print(x.item())     # tensor to num
    
    x = torch.tensor([1, 2])
    y = torch.tensor([2])
    z = x * y
    print(z)
    
def process():
    '''
    torch.max                       返回两个tensor: 最大值和索引
    torch.sum                       返回所有元素的和, dim是要减少的一个或多个维度
    torch.mean                      平局值
    torch.nonzero                   返回非0的索引,返回dimension个tensor,每个tensor的元素表示在该维度上的索引
    torch.eq                        返回一个相同shape的tensor,每个元素为 True or False
    torch.where(condition)          等价于 torch.nonzero
    torch.where(condition, x, y)    返回一个 tensor,condition 为 True 时为 x ,否则为 y
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
    
    print('\n--- sum ---\n')
    print(x.sum())              # 默认减少所有维度
    print(x.sum(dim=0))         # 减少0维
    print(x.sum(dim=1))
    y = torch.tensor([
        [[1, 2, 3],
         [4, 5, 6]],
        [[7, 8, 9],
         [0, 1, 2]]
    ])
    print(y.sum(dim=0))
    
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
    
    x = torch.randn(3, 2)
    y = torch.ones(3, 2)
    print(x)
    print(torch.where(x > 0, x, y))
    

def tensors():
    '''
    多个tensor之间的操作
    stack cat 返回新的tensor,不共享数据
    torch.meshgrid      创建多维网格坐标矩阵
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
    y = torch.tensor([[7,8,9],[-1,-2,-3],[-4,-5,-6]])
    xy = torch.cat((x, y))
    print(xy)
    print(xy.shape)
    y = torch.tensor([[0, -1,-2,-3],[0, -4,-5,-6]])
    xy = torch.cat((x, y), dim=1)
    print(xy)
    print(xy.shape)
    
    a = torch.tensor([1, 2, 3, 4])
    b = torch.tensor([5, 6, 7, 8])
    c = torch.cat((a, b))
    print(c)
    print(c.reshape(-1, 4))
    print(c.reshape(-1, 2))
    
    print('\n<--- meshgrid --->\n')
    
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([1, 2])
    
    xx, yy = torch.meshgrid([x, y], indexing='ij')
    print(xx)
    print(yy)

    
def broadcast():
    x = torch.tensor([1, 2, 3])
    g = torch.tensor(10)
    x += g
    print(x)


def other():
    r"""
    torch.prod          返回所有元素的乘积(product)
    torch.unique        返回所有不同的元素
    """
    print('\n<--- prod --->\n')
    
    x = torch.rand((2, 3, 4))
    print(torch.prod(torch.tensor(x.shape)))
    
    print('\n<--- unique --->\n')
    
    x = torch.tensor([[0, 1, 0, 1, 1, 0, 2, 1]])
    print(x.unique())
    
    
if __name__ == '__main__':
    attribute()
    # test_dtype()
    # change_shape()
    # test_create()
    # base_create()
    # compute()
    # process()
    # tensors()
    # broadcast()
    # other()
