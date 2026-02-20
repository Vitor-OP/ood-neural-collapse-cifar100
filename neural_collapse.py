"""
Neural Collapse visualizations for ResNet CIFAR-100.

Requires features already extracted by run_networks.py.

Produces four figures saved to results/<checkpoint_stem>/:
  nc1_within_class_variance.png  -- per-class within-class variance + global NC1 ratio
  nc2_mean_cosine_similarities.png -- pairwise cosine sim between class means (ETF check)
  nc3_weight_mean_alignment.png  -- cosine sim between W[c] and mu_c per class
  class_mean_distances.png       -- heatmap of pairwise L2 distances between class means
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_networks import extract_weights_and_features
from utils import load_features_from_csv

NUM_CLASSES = 100


def compute_class_means(features, labels):
    """Returns (100, D) array of per-class feature means."""
    means = np.zeros((NUM_CLASSES, features.shape[1]), dtype=np.float64)
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() > 0:
            means[c] = features[mask].mean(axis=0)
    return means


def nc1_within_class_variance(features, labels, class_means):
    """
    Per-class within-class variance (mean squared L2 distance to class mean).
    NC1 ratio: mean(within-class variance) / variance of class means.
    """
    per_class_var = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() > 0:
            diff = features[mask] - class_means[c]
            per_class_var[c] = (diff**2).sum(axis=1).mean()

    global_mean = class_means.mean(axis=0)
    between_class_var = ((class_means - global_mean) ** 2).sum(axis=1).mean()
    nc1_ratio = per_class_var.mean() / (between_class_var + 1e-10)

    return per_class_var, nc1_ratio


def nc2_mean_cosine_similarities(class_means):
    """
    Pairwise cosine similarities between centered class means.
    Under a perfect ETF with C classes, off-diagonal values = -1/(C-1).
    """
    global_mean = class_means.mean(axis=0)
    centered = class_means - global_mean
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / (norms + 1e-10)
    sim_matrix = normalized @ normalized.T  # (100, 100)
    # Extract upper triangle (excluding diagonal)
    idx = np.triu_indices(NUM_CLASSES, k=1)
    off_diag = sim_matrix[idx]
    return sim_matrix, off_diag


def nc3_weight_mean_alignment(weights, class_means):
    """
    Cosine similarity between classifier weight vector W[c] and class mean mu_c
    for each class c. NC3: W[c] / ||W[c]|| ≈ mu_c / ||mu_c|| under neural collapse.
    """
    w_norm = weights / (np.linalg.norm(weights, axis=1, keepdims=True) + 1e-10)
    m_norm = class_means / (np.linalg.norm(class_means, axis=1, keepdims=True) + 1e-10)
    cos_sim = (w_norm * m_norm).sum(axis=1)
    return cos_sim


def plot_nc1(per_class_var, nc1_ratio, save_path):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(NUM_CLASSES), per_class_var, color="steelblue", width=0.8)
    ax.axhline(
        per_class_var.mean(),
        color="crimson",
        linestyle="--",
        label=f"Mean = {per_class_var.mean():.4f}",
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Mean squared distance to class mean")
    ax.set_title(f"NC1 — Within-class variance per class  (NC1 ratio = {nc1_ratio:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}  (NC1 ratio = {nc1_ratio:.6f})")


def plot_nc2(off_diag, save_path):
    etf_ideal = -1.0 / (NUM_CLASSES - 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(off_diag, bins=80, color="steelblue", edgecolor="none")
    ax.axvline(etf_ideal, color="crimson", linestyle="--", label=f"ETF ideal = {etf_ideal:.4f}")
    ax.axvline(
        off_diag.mean(),
        color="orange",
        linestyle="--",
        label=f"Observed mean = {off_diag.mean():.4f}",
    )
    ax.set_xlabel("Pairwise cosine similarity of centered class means")
    ax.set_ylabel("Count")
    ax.set_title("NC2 — Class mean cosine similarities (ETF check)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(
        f"Saved {save_path}  (mean off-diag cos = {off_diag.mean():.4f}, ETF ideal = {etf_ideal:.4f})"
    )


def plot_nc3(cos_sim, save_path):
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["steelblue" if v >= 0 else "crimson" for v in cos_sim]
    ax.bar(range(NUM_CLASSES), cos_sim, color=colors, width=0.8)
    ax.axhline(cos_sim.mean(), color="orange", linestyle="--", label=f"Mean = {cos_sim.mean():.4f}")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, label="Perfect alignment = 1.0")
    ax.set_xlabel("Class")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("NC3 — Alignment between classifier weights and class means")
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}  (mean alignment = {cos_sim.mean():.4f})")


def plot_class_mean_distances(class_means, save_path):
    diff = class_means[:, None, :] - class_means[None, :, :]  # (C, C, D)
    dist_matrix = np.linalg.norm(diff, axis=-1)  # (C, C)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(dist_matrix, cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="L2 distance")
    ax.set_xlabel("Class")
    ax.set_ylabel("Class")
    ax.set_title("Pairwise L2 distances between class means")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    idx = np.triu_indices(NUM_CLASSES, k=1)
    print(f"Saved {save_path}  (mean inter-class dist = {dist_matrix[idx].mean():.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/resnet_cifar_best_best.pth")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    results_dir = checkpoint_path.parent / checkpoint_path.stem
    results_dir.mkdir(exist_ok=True)

    # Load training features (collapse metrics are defined on the training distribution)
    feature_id_train, train_labels = load_features_from_csv(results_dir / "ID_cifar100_train.csv")
    features = np.asarray(feature_id_train, dtype=np.float64)
    labels = np.asarray(train_labels)

    weight, bias, _, _ = extract_weights_and_features(checkpoint_path, None, only_weights=True)
    weight = np.asarray(weight, dtype=np.float64)  # (100, 64)

    # Compute
    class_means = compute_class_means(features, labels)
    per_class_var, nc1_ratio = nc1_within_class_variance(features, labels, class_means)
    _, off_diag = nc2_mean_cosine_similarities(class_means)
    cos_sim = nc3_weight_mean_alignment(weight, class_means)

    # Plot
    plot_nc1(per_class_var, nc1_ratio, results_dir / "nc1_within_class_variance.png")
    plot_nc2(off_diag, results_dir / "nc2_mean_cosine_similarities.png")
    plot_nc3(cos_sim, results_dir / "nc3_weight_mean_alignment.png")
    plot_class_mean_distances(class_means, results_dir / "class_mean_distances.png")
