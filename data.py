from pathlib import Path

import torch
import torchvision
import torchvision.datasets as datasets
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

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=100, shuffle=False)

    test_set = torchvision.datasets.CIFAR100(
        root=root_path / "cifar100", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)
    return train_loader, val_loader, test_loader


def get_cifar100_dataloaders_strong_aug(batch_size=128):
    # Stronger augmentation for best test accuracy
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
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
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_set = torchvision.datasets.CIFAR100(
        root=root_path / "cifar100", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_cifar100_dataloaders_no_aug(batch_size=128):
    # No augmentation — used for collapse mode to reach train error = 0 faster
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.CIFAR100(
        root=root_path / "cifar100", train=True, download=True, transform=transform
    )
    train_size = 45000
    val_size = 5000
    train_subset, val_subset = torch.utils.data.random_split(
        train_set, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    test_set = torchvision.datasets.CIFAR100(
        root=root_path / "cifar100", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_svhn_dataloader(batch_size=100):
    svhn_test = datasets.SVHN(
        root=root_path / "svhn",
        split="test",
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    return torch.utils.data.DataLoader(svhn_test, batch_size=batch_size, shuffle=False)


def get_textures_dataloader(batch_size=100):
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ]
    )
    textures_test = datasets.DTD(
        root=root_path / "textures", split="test", download=True, transform=transform
    )
    return torch.utils.data.DataLoader(textures_test, batch_size=batch_size, shuffle=False)
