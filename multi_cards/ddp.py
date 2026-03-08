import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from torchvision import datasets, transforms
from models.model import get_model, get_cifar10_dataset

def setup(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def main(rank, world_size, args):
    print("rank:", rank, world_size)
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = get_model()
    model.cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare CIFAR-10 dataset
    train_dataset = get_cifar10_dataset(train=True)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    print("train_sampler:", len(train_dataset), args.batch_size, len(train_sampler))
    
    # Create data loader with sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler
    )
    
    # Training loop
    model.train()
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)  # Important for shuffling
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(rank), target.cuda(rank)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            print(f"Rank {rank}: gradient before backward: {model.module.conv1.weight.grad}")
            
            loss.backward()
            
            print(f"Rank {rank}: gradient after backward: {model.module.conv1.weight.grad}")
            
            optimizer.step()
            
            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")
    
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    print("num GPU:", world_size)
    torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
