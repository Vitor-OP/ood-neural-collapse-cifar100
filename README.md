# OOD Detection & Neural Collapse — CIFAR-100

Practical assignment for **5IA23 – Deep Learning Based Computer Vision** at ENSTA Paris.
Trains a ResNet on CIFAR-100 and evaluates OOD detection methods (Softmax, Max Logit, Energy, Mahalanobis, ViM, NECO).

---

## Setup

```bash
pip install torch torchvision scikit-learn scipy tabulate
```

---

## Scripts

### `train.py`

```
python train.py [--model resnet_cifar] [--mode sgd|best|collapse] [--suffix <tag>] [--patience <n>] [--device <cpu|cuda>]
```

| Mode | Optimizer | Augmentation | Stopping |
|------|-----------|--------------|----------|
| `sgd` | SGD lr=0.1, multi-step LR (paper schedule) | crop + hflip | early stop on val plateau |
| `best` | SGD lr=0.1, CosineAnnealingLR T_max=300 | AutoAugment | early stop within 300 epochs |
| `collapse` | Adam lr=1e-3, no WD | none | until train\_acc = 100% |

Saves to `results/resnet_cifar_best_<suffix>.pth` and `results/resnet_cifar_final_<suffix>.pth`.
Checkpoint keys: `epoch`, `model_state_dict`, `optimizer_state_dict`, `train_acc`, `val_acc`, `test_acc`.
Training curves (loss, acc, lr per epoch) saved to `results/train_<timestamp>_resnet_cifar_curves.csv`.

---

### `run_networks.py`

```
python run_networks.py [--checkpoint results/resnet_cifar_best_<suffix>.pth] [--ood svhn textures]
```

Extracts 64-dim penultimate features via a forward hook on `fc` and saves CSVs to `results/<checkpoint_stem>/`:

- `ID_cifar100_train.csv` — 45k training samples
- `ID_cifar100_test.csv` — 10k test samples
- `OOD_svhn_test.csv`, `OOD_textures_test.csv` — OOD test samples

CSV format: columns `ct0..ct63, label`. OOD datasets auto-download on first run.

---

### `ood_methods.py`

```
python ood_methods.py [--checkpoint results/resnet_cifar_best_<suffix>.pth] [--ood svhn textures]
                      [--neco-dim 10] [--latex] [--caption <str>] [--label <str>]
```

Loads CSVs from `results/<checkpoint_stem>/`, recomputes logits as `features @ W.T + b`, and evaluates:

- **Softmax** — max softmax probability
- **Max Logit** — max raw logit
- **Energy** — `logsumexp(logits)`
- **Mahalanobis** — min Mahalanobis distance to class-conditional Gaussians (shared precision, class-centered)
- **ViM** — null-space residual norm + energy; null space = eigenvectors beyond top 75% of feature dims
- **NECO** — fraction of StandardScaler-normalised feature norm captured by the top `--neco-dim` PCA components

Prints AUROC and FPR@95%TPR. Pass `--latex` for LaTeX `tabular` output (side-by-side when two OOD sets).

---

## File Overview

| File | Purpose |
|------|---------|
| `resnet.py` | `ResNet_CIFAR` (3-stack, 64-dim) and `ResNet_ImageNet` architectures |
| `data.py` | Dataloaders: CIFAR-100 (3 variants), SVHN, Textures/DTD |
| `train.py` | Training loop with three modes |
| `run_networks.py` | Feature extraction → CSV |
| `ood_methods.py` | OOD scoring and evaluation |
| `utils.py` | `save_features_to_csv` / `load_features_from_csv` |

---

## Typical Workflow

```bash
# Train
python train.py --mode best --suffix best
python train.py --mode collapse --suffix collapse

# Extract features once per checkpoint
python run_networks.py --checkpoint results/resnet_cifar_best_best.pth --ood svhn textures

# Evaluate
python ood_methods.py --checkpoint results/resnet_cifar_best_best.pth --ood svhn textures
python ood_methods.py --checkpoint results/resnet_cifar_best_best.pth --ood svhn textures --latex
```

---

## Authors

- Vitor Odorissio Pereira
- Rian Radeck

ENSTA Paris
