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
