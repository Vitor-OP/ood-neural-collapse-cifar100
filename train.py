"""
This training script trains a ResNet-18 architecture on CIFAR-100,
with the same training procedure as described in the paper https://arxiv.org/pdf/1512.03385.

But basically, copy paste from the paper:

We use a weight decay of 0.0001 and momentum of 0.9,
and adopt the weight initialization in [13] and BN [16] but
with no dropout. These models are trained with a mini
batch size of 128 on two GPUs. We start with a learning
rate of 0.1, divide it by 10 at 32k and 48k iterations, and
terminate training at 64k iterations, which is determined on
a 45k/5k train/val split. We follow the simple data augmentation
in [24] for training: 4 pixels are padded on each side,
and a 32x32 crop is randomly sampled from the padded
image or its horizontal flip. For testing, we only evaluate
the single view of the original 32x32 image.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_cifar100_dataloaders
from resnet import ResidualBlock, ResNet_CIFAR, ResNet_ImageNet


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def log(message, log_file):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(log_file, "a") as f:
        f.write(log_line + "\n")


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy


def train_model(model_name, device, no_stop=False, best_optim=False):
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    log_file = (
        results_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}_log.txt"
    )

    log(f"Starting training for {model_name}", log_file)
    log(f"Device: {device}", log_file)

    if model_name == "resnet_cifar":
        model = ResNet_CIFAR(ResidualBlock, [2, 2, 2], num_classes=100)
    else:
        model = ResNet_ImageNet(ResidualBlock, [2, 2, 2, 2], num_classes=100)

    model.apply(init_weights)
    model = model.to(device)
    train_loader, val_loader, test_loader = get_cifar100_dataloaders()

    train_dataset_size = len(train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)
    test_dataset_size = len(test_loader.dataset)
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    test_batches = len(test_loader)

    log(f"Train dataset: {train_dataset_size} samples, {train_batches} batches", log_file)
    log(f"Val dataset: {val_dataset_size} samples, {val_batches} batches", log_file)
    log(f"Test dataset: {test_dataset_size} samples, {test_batches} batches", log_file)

    criterion = nn.CrossEntropyLoss()

    if best_optim:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        log("Using AdamW optimizer with lr=1e-3", log_file)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        log("Using SGD optimizer with lr=0.1", log_file)

    max_iterations = 64000
    lr_decay_iterations = [32000, 48000]
    lr_decay_epochs = [it // train_batches for it in lr_decay_iterations]
    max_epochs = max_iterations // train_batches
    if best_optim:
        max_epochs = 2400

    if best_optim:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_epochs, gamma=0.1)

    log(
        (
            f"Training for ~{max_iterations} iterations ({max_epochs} epochs)"
            f", LR decay at epochs {lr_decay_epochs}"
        ),
        log_file,
    )

    start_time = time.time()
    best_test_acc = 0
    epoch = 0
    train_error_zero = False

    while (
        not train_error_zero
        and (epoch < max_epochs or no_stop)
        and (time.time() - start_time < int(3.8 * 3600))
    ):
        epoch += 1
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        train_loss = train_loss / len(train_loader)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        log(
            (
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% |"
                f" Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% |"
                f" LR: {optimizer.param_groups[0]['lr']:.6f} | Elapsed: {elapsed_str}"
            ),
            log_file,
        )

        if train_acc == 100.0:
            train_error_zero = True
            log("TPT reached (train error = 0)", log_file)

        if val_acc > best_test_acc:
            best_test_acc = val_acc
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
            }
            save_path = results_dir / f"{model_name}_best.pth"
            torch.save(checkpoint, save_path)
            log(
                f"Saved best model with val acc: {val_acc:.2f}%, test acc: {test_acc:.2f}%",
                log_file,
            )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    final_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
    }
    final_path = results_dir / f"{model_name}_final.pth"
    torch.save(final_checkpoint, final_path)
    total_time = time.time() - start_time
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
    log(
        (
            f"Training completed. Final val acc: {val_acc:.2f}%, test acc: {test_acc:.2f}% |"
            f" Total time: {total_time_str}"
        ),
        log_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["resnet_cifar", "resnet_18"], required=True)
    parser.add_argument(
        "--no-stop", action="store_true", help="Don't stop training when max epochs is reached"
    )
    parser.add_argument(
        "--best-optim",
        action="store_true",
        help="Change SGD to a better optim for faster convergence",
    )

    # added only for dev purposes
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    train_model(args.model, args.device, no_stop=args.no_stop, best_optim=args.best_optim)
