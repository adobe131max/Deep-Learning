def test_list():
    pass


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
        
    for key, value in info.items():
        print(key, value)
        

if __name__ == '__main__':
    test_tuple()
