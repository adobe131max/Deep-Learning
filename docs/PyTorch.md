# PyTorch

## Tensor

## 混合精度

结合半精度（FP16）与单精度（FP32）训练网络，减少训练时间，降低内存需求

1. `torch.cuda.amp.autocast()` 将模型的前向传播和损失计算转换为FP16格式
2. `scaler.scale(loss).backward()` 梯度缩放有助于防止训练时使用混合精度时，具有小幅度的梯度被冲刷为零（“下溢”），通过放大损失并在反向传播时相应缩小梯度，防止因二进制表示误差导致的训练问题，即先对loss进行缩放再反向传播计算梯度，取代 `loss.backward()`
3. `scaler.step(optimizer)` 更新参数，取代 `optimizer.step()`
4. `scaler.update()` 更新缩放因子

``` py
import torch

scaler = torch.cuda.amp.GradScaler()

# one epoch example
for inputs, targets in dataloader:
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# PyTorch 2.4 以后的版本使用：
scaler = torch.amp.GradScaler()     # 

for inputs, targets in dataloader:
    optimizer.zero_grad()
    
    with torch.amp.autocast('cuda'):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

```

## 多机多卡

## TODO

1. BF16