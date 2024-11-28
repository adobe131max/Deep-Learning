def test_list():
    l1 = [1, 2, 3]
    l2 = [4, 5, 6]
    # 创建新列表
    l3 = l1 + l2
    print(l3)
    # 直接修改原列表
    l1.extend(l2)
    print(l1)
    l1.append(7)
    print(l1)


def test_tuple():
    tp = ([1], [2], [3])
    # 不能直接修改元组中的元素
    tp[0] = [0]
    print(tp)
    # but this is ok
    tp[0][0] = 0
    print(tp)


def test_dict():
    info = {
        "name": "John",
        "age": 30,
        "city": "New York"
    }

    for key in info:
        print(key)
    
    for key in info.keys():
        print(key)
    
    for value in info.values():
        print(value)
        
    for key, value in info.items():
        print(key, value)
    
    # 字典推导式
    numbers = [1, 2, 3, 4, 5]
    squares = {n: n**2 for n in numbers}

# TODO: defaultdict

def test_set():
    pass
        

if __name__ == '__main__':
    # test_list()
    # test_tuple()
    test_dict()
