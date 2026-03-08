import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from torchvision import datasets, transforms
import deepspeed
from torch.utils.data import DataLoader
from models.model import get_model, get_cifar10_dataset


def create_deepspeed_config():
    """Create DeepSpeed configuration for ZeRO-1 optimization"""
    return {
        "train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "具体",
            "params": {
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0.0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 100
            }
        },
        "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "wall_clock_breakdown": False
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank passed by DeepSpeed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()

    # Initialize DeepSpeed
    deepspeed.init_distributed()

    # Create model
    model = get_model()

    # Prepare CIFAR-10 dataset
    train_dataset = get_cifar10_dataset(train=True)

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # DeepSpeed handles shuffling
        num_workers=4,
        pin_memory=True
    )

    # Create DeepSpeed configuration
    ds_config = create_deepspeed_config()
    ds_config["train_batch_size"] = args.batch_size

    # Initialize DeepSpeed model and optimizer
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # Training loop
    model_engine.train()
    for epoch in range(args.num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device (DeepSpeed handles device placement)
            print("model_engine.local_rank:", model_engine.local_rank)
            data, target = data.to(model_engine.local_rank), target.to(model_engine.local_rank)

            # Forward pass
            optimizer.zero_grad()
            output = model_engine(data)
            loss = model_engine.criterion(output, target) if hasattr(model_engine, 'criterion') else nn.CrossEntropyLoss()(output, target)

            # Backward pass (DeepSpeed handles gradient accumulation and synchronization)
            model_engine.backward(loss)
            model_engine.step()

            # Logging
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item():.4f}")

        # Save checkpoint at end of epoch
        if model_engine.global_rank == 0:
            model_engine.save_checkpoint(f"./checkpoint-epoch-{epoch}")


if __name__ == "__main__":
    main()