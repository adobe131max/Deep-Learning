import os
import sys
import math
import torch
import tempfile
import torch.distributed as dist

from tqdm import tqdm
from torchvision.models import resnet50


def setup(rank, local_rank, world_size):
    # 设置当前进程所使用的 GPU 设备
    # CUDA_VISIBLE_DEVICES 重新映射了使用的 GPU 编号，这里的 local_rank 对应的实际是 CUDA_VISIBLE_DEVICES[local_rank]
    # 比如设置 CUDA_VISIBLE_DEVICES=2,3 local_rank 0 对应的实际是 GPU 2
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
    
    
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
    
    for inputs, targets in tqdm(dataloader, ncols=50, position=dist.get_rank()):
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
            
        dist.all_reduce(loss, dist.ReduceOp.AVG)
            
        epoch_loss += loss.item()
    
    lr = optimizer.param_groups[0]["lr"]
    
    lr_scheduler.step()
    
    epoch_loss /= len(dataloader)   # 单卡上的迭代的次数

    return epoch_loss, lr


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    
    epoch_loss = 0.0
    
    for inputs, targets in tqdm(dataloader, ncols=50, position=dist.get_rank()):
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        dist.all_reduce(loss, dist.ReduceOp.AVG)
        
        epoch_loss += loss.item()
        
    epoch_loss /= len(dataloader)
    
    return epoch_loss


def train():
    rank       = int(os.environ['RANK'])        # dist.get_ranl()
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])  # dist.get_world_size()
    
    setup(rank, local_rank, world_size)
    dist.barrier()
    
    device = torch.device('cuda')
    best_loss = 10.0
    batch_size = 16
    epochs = 100
    lr = 1e-4
    
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    if sys.platform.startswith('win'):
        num_workers = 0
    
    train_dataset = None
    val_dataset   = None
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=None,
        pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=None,
        pin_memory=True,
    )
    
    model = resnet50().cuda()
    
    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        torch.save(model.state_dict(), CHECKPOINT_PATH)
    
    dist.barrier()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = create_lr_scheduler(optimizer, epochs)
    scaler = torch.amp.GradScaler()
    
    if rank == 0:
        pass
    
    for epoch in range(1, epochs + 1):
        dist.barrier()
        train_sampler.set_epoch(epoch)
        train_loss, lr = train_one_epoch(model, optimizer, lr_scheduler, train_dataloader, scaler)
        
        dist.barrier()
        if rank == 0:
            pass
        
        dist.barrier()
        val_loss = evaluate(model, val_dataloader, local_rank)
        
        dist.barrier()
        if rank == 0:
            if epoch > 10 and val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.module.state_dict(), f'epoch{epoch}_{val_loss:.4f}.pth')
    
    if rank == 0:
        os.remove(CHECKPOINT_PATH)
    
    cleanup()
    

if  __name__ == '__main__':
    train()
    