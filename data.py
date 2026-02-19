from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

root_path = Path(__file__).parent


def get_cifar100_dataloaders():
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.CIFAR100(
        root=root_path / "cifar100", train=True, download=True, transform=transform_train
    )
    train_size = 45000
    val_size = 5000
    train_subset, val_subset = torch.utils.data.random_split(
        train_set, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=100, shuffle=False)

    test_set = torchvision.datasets.CIFAR100(
        root=root_path / "cifar100", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)
    return train_loader, val_loader, test_loader
