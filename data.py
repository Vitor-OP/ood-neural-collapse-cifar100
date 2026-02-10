import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

root_path = Path(__file__).parent

def get_cifar100_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.CIFAR100(
        root=root_path / "cifar100", train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

    test_set = torchvision.datasets.CIFAR100(
        root=root_path / "cifar100", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)
    return train_loader, test_loader
