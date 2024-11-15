import torch

w = torch.tensor(10.0, requires_grad=True)
h = torch.tensor(1.0, requires_grad=True)

def model(inp):
    return w * inp + h

def criterion(a, b):
    return (a - b) ** 2


def test1():
    x = torch.tensor(2.0)
    z = torch.tensor(4.0)

    y = model(x)
    loss = criterion(z, y)

    print(loss.dtype)
    print(loss)


def test2():
    scaler = torch.cuda.amp.GradScaler()
    
    x = torch.tensor(2.0)
    z = torch.tensor(4.0)

    with torch.cuda.amp.autocast():
        y = model(x)
        loss = criterion(z, y)
    
    print(loss.dtype)
    print(loss)

    loss = scaler.scale(loss)
    
    print(loss.dtype)
    print(loss)
    

if __name__ == '__main__':
    # test1()
    test2()
