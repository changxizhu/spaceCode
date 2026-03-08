import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from accelerate import Accelerator
from models.model import get_model, get_cifar10_dataset


def train_epoch(model, train_loader, optimizer, criterion, accelerator):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass (handled by accelerator)
        accelerator.backward(loss)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            accelerator.print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, val_loader, criterion, accelerator):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in val_loader:
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()

    # Create model
    model = get_model()
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare datasets
    train_dataset = get_cifar10_dataset(train=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Accelerate handles shuffling for distributed training
        num_workers=4,
        pin_memory=True
    )
    
    # Prepare model, optimizer, and dataloader with Accelerator
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )
    
    # Training loop
    accelerator.print(f"Starting training on {accelerator.device}")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    
    for epoch in range(args.num_epochs):
        accelerator.print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, accelerator)
        accelerator.print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if accelerator.is_main_process:
            checkpoint_path = f"./checkpoint-epoch-{epoch}.pt"
            accelerator.save_state(f"./checkpoint-epoch-{epoch}")
            accelerator.print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
