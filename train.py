from data import get_cifar100_dataloaders
from resnet import ResNet_CIFAR, ResidualBlock

resnet_cifar = ResNet_CIFAR(ResidualBlock, [2, 2, 2], num_classes=100)

if __name__ == "__main__":
    train_loader, test_loader = get_cifar100_dataloaders()
    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
        break