import os
import sys
import math
import torch

from tqdm import tqdm
from torchvision.models import resnet50


def create_lr_scheduler(
    optimizer,
    epochs: int,
    warmup=False,
    warmup_epochs=1,
    warmup_factor=1e-2,
    end_factor=1e-5
):
    assert epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= warmup_epochs:
            alpha = float(x) / warmup_epochs
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = x - warmup_epochs
            cosine_steps = epochs - warmup_epochs
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def criterion(predicts, targets):
    pass


def train_one_epoch(model, optimizer, lr_scheduler, dataloader, scaler=None):
    model.train()
    
    epoch_loss = 0.0
    
    for inputs, targets in tqdm(dataloader, ncols=50):
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        epoch_loss += loss.item() * len(inputs)
    
    lr = optimizer.param_groups[0]["lr"]
    
    lr_scheduler.step()
    
    epoch_loss /= len(dataloader.dataset)

    return epoch_loss, lr


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    
    epoch_loss = 0.0
    
    for inputs, targets in tqdm(dataloader, ncols=50):
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        epoch_loss += loss.item() * len(inputs)
        
    epoch_loss /= len(dataloader.dataset)
    
    return epoch_loss


def train():
    torch.cuda.set_device(0)
    best_loss = 10.0
    batch_size = 16
    epochs = 100
    lr = 1e-4
    
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    if sys.platform.startswith('win'):
        num_workers = 0
    
    train_dataset = None
    val_dataset   = None
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=None,
        pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=None,
        pin_memory=True,
    )
    
    model = resnet50().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = create_lr_scheduler(optimizer, epochs)
    scaler = torch.amp.GradScaler()
    
    for epoch in range(1, epochs + 1):
        
        train_loss, lr = train_one_epoch(model, optimizer, lr_scheduler, train_dataloader, scaler)
        
        print('...')
        
        val_loss = evaluate(model, val_dataloader)
        
        print('...')
        
        if epoch > 10 and val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.module.state_dict(), f'epoch{epoch}_{val_loss:.4f}.pth')


if __name__ == '__main__':
    train()
