import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data import (
    get_cifar100_dataloaders,
    get_cifar100_dataloaders_no_aug,
    get_cifar100_dataloaders_strong_aug,
)
from resnet import ResidualBlock, ResNet_CIFAR


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


def train_model(model_name, device, mode="sgd", suffix="", patience=20):
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    log_file = (
        results_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}_log.txt"
    )

    curves_file = (
        results_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}_curves.csv"
    )
    curves_fh = open(curves_file, "w", newline="")
    curves_writer = csv.DictWriter(
        curves_fh, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
    )
    curves_writer.writeheader()

    log(f"Starting training | model={model_name} | mode={mode}", log_file)
    log(f"Device: {device}", log_file)

    model = ResNet_CIFAR(ResidualBlock, [2, 2, 2], num_classes=100)
    model.apply(init_weights)
    model = model.to(device)

    if mode == "collapse":
        # No augmentation — model sees same images every epoch, easiest path to train error = 0
        train_loader, val_loader, test_loader = get_cifar100_dataloaders_no_aug()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
        scheduler = None
        max_epochs = 10000  # run until train error = 0
        log("Collapse mode: no augmentation, Adam, no weight decay", log_file)
    elif mode == "best":
        # Strong augmentation + SGD + cosine annealing — maximizes test accuracy
        train_loader, val_loader, test_loader = get_cifar100_dataloaders_strong_aug()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        max_epochs = 1000
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        log(f"Best mode: strong aug, SGD + cosine LR, early stop patience={patience}", log_file)
    else:
        # SGD with paper schedule + early stopping on val plateau
        train_loader, val_loader, test_loader = get_cifar100_dataloaders()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        train_batches = len(train_loader)
        lr_decay_iterations = [32000 * 2, 48000 * 2]
        lr_decay_epochs = [it // train_batches for it in lr_decay_iterations]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_epochs, gamma=0.1)
        max_epochs = 10000
        log(
            f"SGD mode: LR decay at epochs {lr_decay_epochs}, early stop patience={patience}",
            log_file,
        )

    criterion = nn.CrossEntropyLoss()

    log(
        f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}",
        log_file,
    )

    start_time = time.time()
    best_val_acc = 0
    epochs_no_improve = 0
    epoch = 0

    while epoch < max_epochs and (time.time() - start_time < int(3.95 * 3600)):
        epoch += 1
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
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

        if scheduler:
            scheduler.step()

        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        lr = optimizer.param_groups[0]["lr"]
        log(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% |"
            f" Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% |"
            f" LR: {lr:.6f} | Elapsed: {elapsed_str}",
            log_file,
        )
        curves_writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": lr,
            }
        )
        curves_fh.flush()

        # Save best val checkpoint (SGD mode only)
        if (mode == "sgd" or mode == "best") and val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            _, test_acc = evaluate(model, test_loader, criterion, device)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                },
                results_dir / f"{model_name}_best_{suffix}.pth",
            )
            log(f"Saved best model | val_acc={val_acc:.2f}% | test_acc={test_acc:.2f}%", log_file)
        elif mode == "sgd":
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log(f"Early stop: no val improvement for {patience} epochs", log_file)
                break

        # Collapse mode: stop when train error = 0
        if train_acc == 100.0:
            log("Train error = 0 reached (neural collapse state)", log_file)
            break

    curves_fh.close()
    log(f"Training curves saved to {curves_file}", log_file)

    # Save final checkpoint
    _, test_acc = evaluate(model, test_loader, criterion, device)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
        },
        results_dir / f"{model_name}_final_{suffix}.pth",
    )
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    log(
        f"Done | val_acc={val_acc:.2f}% | test_acc={test_acc:.2f}% | time={total_time_str}",
        log_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet_cifar")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sgd", "collapse", "best"],
        default="sgd",
        help="sgd: paper schedule + early stop | best: strong aug + cosine LR | collapse: drive train error to 0",
    )
    parser.add_argument("--patience", type=int, default=30, help="Early stop patience (sgd mode)")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    train_model(args.model, args.device, mode=args.mode, suffix=args.suffix, patience=args.patience)
