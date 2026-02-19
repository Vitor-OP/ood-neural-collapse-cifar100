# OOD Detection & Neural Collapse — CIFAR-100

Practical assignment for **5IA23 – Deep Learning Based Computer Vision** at ENSTA Paris.
Trains a ResNet on CIFAR-100 and evaluates OOD detection methods.

---

## Setup

```bash
pip install torch torchvision scikit-learn scipy
```

---

## Scripts

### `train.py` — Train the model
```bash
python train.py --model resnet_cifar [--best-optim] [--no-stop] [--suffix <tag>]
```
- `resnet_cifar`: 3-stack ResNet adapted for 32×32 CIFAR input
- `--best-optim`: use AdamW + cosine LR instead of SGD
- Saves to `results/resnet_cifar_best_<suffix>.pth` and `results/resnet_cifar_final_<suffix>.pth`
- Checkpoint keys: `model_state_dict`, `optimizer_state_dict`, `train_acc`, `val_acc`, `test_acc`, `epoch`

---

### `run_networks.py` — Extract and save features to CSV
```bash
python run_networks.py [--checkpoint results/resnet_cifar_best_adam.pth] [--ood svhn textures]
```
- Runs the model on train/test ID (CIFAR-100) and OOD dataloaders
- Extracts penultimate layer features (64-dim) via a forward hook on `fc`
- Saves CSVs to `results/<checkpoint_stem>/`:
  - `ID_cifar100_train.csv` — 45k train samples, columns `ct0..ct63, label`
  - `ID_cifar100_test.csv` — 10k test samples
  - `OOD_<name>_test.csv` — OOD test samples (SVHN and/or Textures)
- OOD datasets are auto-downloaded on first run

---

### `ood_methods.py` — Evaluate OOD detection methods
```bash
python ood_methods.py [--checkpoint results/resnet_cifar_best_adam.pth] [--ood svhn textures]
```
- Loads CSVs produced by `run_networks.py`
- Computes logits as `features @ W.T + b` and runs five scoring methods:
  - **Softmax** — max softmax probability
  - **Max Logit** — max raw logit
  - **Energy** — `-logsumexp(logits)`
  - **Mahalanobis** — min distance to class-conditional Gaussians (shared precision matrix)
  - **ViM** — residual norm in null space of feature covariance + energy
- Prints AUROC and FPR@95%TPR per method per OOD dataset

---

## File Overview

| File | Purpose |
|------|---------|
| `resnet.py` | `ResNet_CIFAR` and `ResNet_ImageNet` architectures |
| `data.py` | Dataloaders for CIFAR-100, SVHN, Textures |
| `train.py` | Training loop |
| `run_networks.py` | Feature extraction → CSV |
| `ood_methods.py` | OOD scoring and evaluation |
| `utils.py` | CSV save/load helpers |

---

## Typical Workflow

```bash
# 1. Train
python train.py --model resnet_cifar --best-optim --suffix adam

# 2. Extract features (downloads OOD datasets automatically)
python run_networks.py --ood svhn textures

# 3. Evaluate
python ood_methods.py --ood svhn textures
```

---

## Authors

- Vitor Odorissio Pereira
- Rian Radeck

ENSTA Paris
