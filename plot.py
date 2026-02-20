import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(csv_file, sample_interval=10):
    """
    Plot training curves from a CSV file.

    Parameters:
    csv_file: Path to the CSV file containing training curves
    sample_interval: Sample every N epochs (default: 10)
    """
    df = pd.read_csv(csv_file)

    # Sample every N epochs
    df_sampled = df[df["epoch"] % sample_interval == 0].copy()

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Loss
    axes[0].plot(df_sampled["epoch"], df_sampled["train_loss"], label="Train Loss", linewidth=1.5)
    axes[0].plot(df_sampled["epoch"], df_sampled["val_loss"], label="Val Loss", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot Accuracy
    axes[1].plot(
        df_sampled["epoch"], df_sampled["train_acc"], label="Train Accuracy", linewidth=1.5
    )
    axes[1].plot(df_sampled["epoch"], df_sampled["val_acc"], label="Val Accuracy", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path(csv_file).parent / f"{Path(csv_file).stem}_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curves from CSV file")
    parser.add_argument(
        "--csv",
        type=str,
        default="results/train_20260219_235237_resnet_cifar_curves.csv",
        help="Path to the CSV file containing training curves",
    )
    args = parser.parse_args()

    csv_path = Path(__file__).parent / args.csv

    if not csv_path.exists():
        print(f"Error: File {csv_path} not found")
        exit(1)

    plot_training_curves(csv_path)
