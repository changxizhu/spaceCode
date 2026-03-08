import torch
import torch.nn as nn
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    """Simple CNN model for CIFAR-10 classification"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model():
    """Factory function to create the model"""
    return SimpleCNN()


def get_cifar10_transform():
    """Get the standard CIFAR-10 data transformation"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_cifar10_dataset(train=True, transform=None):
    """Get CIFAR-10 dataset with optional transform"""
    if transform is None:
        transform = get_cifar10_transform()

    return datasets.CIFAR10(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )
